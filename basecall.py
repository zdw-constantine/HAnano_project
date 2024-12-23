import argparse

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.cuda import get_device_capability
import datetime
from time import perf_counter
from datetime import timedelta

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from _runtime import lib, ffi
from koi.utils import void_ptr, empty, zeros


from bonito_module.reader import Reader

from bonito_module.bonito_io import biofmt,Writer
from bonito_module.bonito_utils import load_model,load_symbol,column_to_set
from bonito_module.multiprocessing import process_cancel,thread_iter



def chunk(signal, chunksize, overlap):
    """
    Convert a read into overlapping chunks before calling
    """
    if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal)

    T = signal.shape[0]
    if chunksize == 0:
        chunks = signal[None, :]
    elif T < chunksize:
        chunks = torch.nn.functional.pad(signal, (chunksize - T, 0))[None, :]
    else:
        stub = (T - overlap) % (chunksize - overlap)
        chunks = signal[stub:].unfold(0, chunksize, chunksize - overlap)
    
    return chunks.unsqueeze(1)

'''
def chunk(signal, chunksize, overlap):
    """
    Convert a read into overlapping chunks before calling
    """
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)
    T = signal.shape[-1]
    if chunksize == 0:
        chunks = signal[None, :]
    elif T < chunksize:
        n, overhang = divmod(chunksize, T)
        # np.tile operates only on dimension -1 by default, 
        # whereas torch.repeat requires explicit listing of all input dimensions eg (1,n) or (1,1,n)
        chunks = torch.cat((torch.from_numpy(np.tile(signal,n)), signal[...,:overhang]), dim=-1)[None, :]
    else:
        stub = (T - overlap) % (chunksize - overlap)
        chunks = signal[...,stub:].unfold(-1, chunksize, chunksize - overlap).movedim(-2,0)
        if stub > 0:
            chunks = torch.cat([signal[None, ..., :chunksize], chunks], dim=0)
    
    return chunks
'''

def size(x, dim=0):
    """
    Type agnostic size.
    """
    if hasattr(x, 'shape'):
        return x.shape[dim]
    elif dim == 0:
        return len(x)
    raise TypeError

def concat(xs, dim=0):
    """
    Type agnostic concat.
    """
    if isinstance(xs[0], torch.Tensor):
        return torch.cat(xs, dim=dim)
    elif isinstance(xs[0], np.ndarray):
        return np.concatenate(xs, axis=dim)
    elif isinstance(xs[0], list):
        return [x for l in xs for x in l]
    elif isinstance(xs[0], str):
        return ''.join(xs)
    elif isinstance(xs[0], dict):
        return {k: concat([x[k] for x in xs], dim) for k in xs[0].keys()}
    else:
        raise TypeError


def select_range(x, start, end, dim=0):
    """
    Type agnostic range select.
    """
    if isinstance(x, dict):
        return {k: select_range(v, start, end, dim) for (k, v) in x.items()}
    if dim == 0 or isinstance(x, list): return x[start:end]
    return x[(*(slice(None),)*dim, slice(start, end))]

def batchify(items, batchsize, dim=0):
    """
    Batch up items up to `batch_size`.
    """
    stack, pos = [], 0
    for k, v in items:
        breaks = range(batchsize - pos, size(v, dim), batchsize)
        for start, end in zip([0, *breaks], [*breaks, size(v, dim)]):
            sub_batch = select_range(v, start, end, dim)
            stack.append(((k, (pos, pos + end - start)), sub_batch))
            if pos + end - start == batchsize:
                ks, vs = zip(*stack)
                yield ks, concat(vs, dim)
                stack, pos = [], 0
            else:
                pos += end - start

    if len(stack):
        ks, vs = zip(*stack)
        yield ks, concat(vs, dim)

def half_supported():
    """
    Returns whether FP16 is support on the GPU
    """
    try:
        return get_device_capability()[0] >= 7
    except:
        return False

def beam_search(scores, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0, move_pad=True):

    if scores.dtype != torch.float16:
        raise TypeError('Expected fp16 but received %s' % scores.dtype)

    assert(scores.is_contiguous())

    N, T, C =  scores.shape

    chunks = torch.empty((N, 4), device=scores.device, dtype=torch.int32)
    chunks[:, 0] = torch.arange(0, T * N, T)
    chunks[:, 2] = torch.arange(0, T * N, T)
    chunks[:, 1] = T
    chunks[:, 3] = 0
    chunk_results = empty((N, 8), device=scores.device, dtype=torch.int32)

    # todo: reuse scores buffer?
    aux      = empty(N * (T + 1) * (C + 4 * beam_width), device=scores.device, dtype=torch.int8)
    path     = zeros(N * (T + 1), device=scores.device, dtype=torch.int32)

    moves    = zeros(N * T, device=scores.device, dtype=torch.int8)
    sequence = zeros(N * T, device=scores.device, dtype=torch.int8)
    qstring  = zeros(N * T, device=scores.device, dtype=torch.int8)

    args = [
        void_ptr(chunks),
        chunk_results.ptr,
        N,
        void_ptr(scores),
        C,
        aux.ptr,
        path.ptr,
        moves.ptr,
        ffi.NULL,
        sequence.ptr,
        qstring.ptr,
        scale,
        offset,
        beam_width,
        beam_cut,
        blank_score,
    ]

    lib.host_back_guide_step(*args)
    lib.host_beam_search_step(*args)
    lib.host_compute_posts_step(*args)
    lib.host_run_decode(*args, int(move_pad))

    moves_ = moves.data.reshape(N, -1).cpu()
    sequence_ = sequence.data.reshape(N, -1).cpu()
    qstring_ = qstring.data.reshape(N, -1).cpu()

    return sequence_, qstring_, moves_

def compute_scores(model, batch, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0, reverse=False):
    """
        Compute scores for model.
    """
    #sys.stderr.write(f"<batch1>  \n")
    with torch.inference_mode():
    
        #print("<batch>")
        device = next(model.parameters()).device
        #dtype = torch.float16 if half_supported() else torch.float32
        #sys.stderr.write(f"<batch2>  \n")

        
        #print(batch)
        with torch.cuda.amp.autocast(enabled=model.use_amp):
            p = model.forward(batch.to(device))
        #print(p.shape)    
        scores = p.cuda().to(torch.float32)
        
        betas = model.seqdist.backward_scores(scores.to(torch.float32))
        trans, init = model.seqdist.compute_transition_probs(scores, betas)
        trans = trans.to(torch.float32).transpose(0, 1)
        init = init.to(torch.float32).unsqueeze(1)
        init = init[0, 0]
        #print("sssssssssssssssssssssss")
        #print(trans)
        #print(init)
        
        '''stacked_transitions = model.stitch_by_stride(
                chunks = np.vstack(trans), 
                chunksize = 2000, 
                overlap = 400, 
                #length = batch['len'].squeeze(0)[0].item(), 
                stride = False
        )'''
        #scores = model(batch.to(dtype).to(device))
        
        #if reverse:
        #    scores = model.seqdist.reverse_complement(scores)

        
        '''sequence, qstring, moves = beam_search(
            scores, beam_width=beam_width, beam_cut=beam_cut,
            scale=scale, offset=offset, blank_score=blank_score
        )'''
            
        return {
            'trans':trans,
            'init': init,
        }
        #return batch

def stitch_by_stride( chunks, chunksize, overlap, length, stride, reverse=False):

        # print("-----------------------------------------")        
        # print(chunksize)

        # print(overlap)
        # print(length)
        # print(stride)
        # print(reverse)
        """
        Stitch chunks together with a given overlap
        
        This works by calculating what the overlap should be between two outputed
        chunks from the network based on the stride and overlap of the inital chunks.
        The overlap section is divided in half and the outer parts of the overlap
        are discarded and the chunks are concatenated. There is no alignment.
        
        Chunk1: AAAAAAAAAAAAAABBBBBCCCCC
        Chunk2:               DDDDDEEEEEFFFFFFFFFFFFFF
        Result: AAAAAAAAAAAAAABBBBBEEEEEFFFFFFFFFFFFFF
        
        Args:
            chunks (tensor): predictions with shape [samples, length, *]
            chunk_size (int): initial size of the chunks
            overlap (int): initial overlap of the chunks
            length (int): original length of the signal
            stride (int): stride of the model
            reverse (bool): if the chunks are in reverse order
            
        Copied from https://github.com/nanoporetech/bonito
        """

        if isinstance(chunks, np.ndarray):
            chunks = torch.from_numpy(chunks)

        if chunks.shape[0] == 1: return chunks.squeeze(0)

        semi_overlap = overlap // 2
        start, end = semi_overlap // stride, (chunksize - semi_overlap) // stride
        stub = (length - overlap) % (chunksize - overlap)
        first_chunk_end = (stub + semi_overlap) // stride if (stub > 0) else end

        if reverse:
            chunks = list(chunks)
            return torch.cat([
                chunks[-1][:-start], *(x[-end:-start] for x in reversed(chunks[1:-1])), chunks[0][-first_chunk_end:]
            ])
        else:
            return torch.cat([
                chunks[0, :first_chunk_end], *chunks[1:-1, start:end], chunks[-1, start:]
            ])
  

def compute_scores2(model,read_key,read_value, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0, reverse=False):
    """
        Compute scores for model.
    """
    with torch.inference_mode():
        #print(read_key[0].read_id)
        #print(read_value.shape)

        transition_scores = list()
        for x in read_value:
            
            device = next(model.parameters()).device
            #dtype = torch.float16 if half_supported() else torch.float32
            #sys.stderr.write(f"<batch2>  \n")

            
            #print(batch)
            
            #x=x.squeeze(1)
            #print(x.shape)
            with torch.cuda.amp.autocast(enabled=model.use_amp):
                p = model.forward(x.to(device))
            #print(p.shape)    
            scores = p.cuda().to(torch.float32)
            #print(scores.shape)
            betas = model.seqdist.backward_scores(scores.to(torch.float32))
            trans, init_ = model.seqdist.compute_transition_probs(scores, betas)
            trans = trans.to(torch.float32).transpose(0, 1)
            init = init_.to(torch.float32).unsqueeze(1)

            #---------------------------------------------------------------------------------------------------------------------------------------------
            #print(p.shape)
            #scores = model.compute_scores(p, use_fastctc=True)
            #print(scores[0].shape)
            transition_scores.append(trans.cpu())
            #print(trans.shape)
        init = init[0, 0].cpu()

        #print(len(transition_scores))
        #print(init.shape)


        stacked_transitions = stitch_by_stride(
                chunks = np.vstack(transition_scores), 
                chunksize = 2000, 
                overlap = 400, 
                length = read_key[2], 
                stride = 5
        )
        #print("stacked_transitions")
        #print(stacked_transitions[0])

        
        seq, path = model._decode_crf_greedy_fastctc(
                tracebacks = stacked_transitions.numpy(), 
                init = init.numpy(), 
                qstring = True, 
                qscale =1.0, 
                qbias = 1.0,
                return_path = True,
                
        )
        
        # fastq_string = '@'+str(read_key[0].read_id)+'\n'
        # fastq_string += seq[:len(path)] + '\n'
        # fastq_string += '+\n'
        # fastq_string += seq[len(path):] + '\n'

        # print(fastq_string)
        
        return {
            'stride': 5,
            'moves': 0,
            'qstring': seq[len(path):],
            'sequence': seq[:len(path)],
        }
    

def basecall(model, reads, chunksize=2000, overlap=400, batchsize=32,
             reverse=False, rna=False):
    """
    Basecalls a set of reads.
    """
    
    chunks = thread_iter(
        ((read, 0, read.signal.shape[-1]), chunk(torch.from_numpy(read.signal), chunksize, overlap))
        for read in reads
    )

    

    batches = thread_iter(batchify(chunks, batchsize=batchsize))

    #print(list(batches))

    read_batches_dict={}
    for batch in batches:
        for batch_read in batch[0]:
            batch_key=batch_read[0]
            #print(batch_read[1])
            if batch_key not in read_batches_dict:
                read_batches_dict[batch_key]=[]
                read_batches_dict[batch_key].append(batch[1][batch_read[1][0]:batch_read[1][1]])
            else:
                read_batches_dict[batch_key].append(batch[1][batch_read[1][0]:batch_read[1][1]])

    # for i in read_batches_dict.values():
    #     for j in i:
    #         print(j.shape)

    '''for read_key in read_batches_dict.keys():
        #print(read)    (((Read('0d2d93b4-3da7-4508-8f86-a58497983b10'), 0, 110482), (0, 32)),)
        compute_scores2(model,read_key,read_batches_dict[read_key], reverse=reverse)'''

   

    #print(list(result))

    '''for read, batch in batches:
        #print(read)    (((Read('0d2d93b4-3da7-4508-8f86-a58497983b10'), 0, 110482), (0, 32)),)
        compute_scores(model, batch, reverse=reverse)
    '''
        
    '''scores = thread_iter(
        (read, compute_scores(model, batch, reverse=reverse)) for read, batch in batches
    )'''
    #print(list(scores))


    '''read_scores_dict={}
    for read in scores:
        list_aaa=[]
        #print(read[0]) #i[0] read-1,read-2
        for read_part in read[0]:
            #print(read_part)
            #print(read_part[0])
            
            if  read_part[0] not in read_scores_dict:
                #print(read[1]['trans'][read_part[1][0]:read_part[1][1]].shape)
                read_scores_dict[read_part[0]]=[[],[]]
                read_scores_dict[read_part[0]][0].append(read[1]['trans'][read_part[1][0]:read_part[1][1]])
                read_scores_dict[read_part[0]][1].append(read[1]['init'][read_part[1][0]:read_part[1][1]])
            else:
                #print(read[1]['trans'][read_part[1][0]:read_part[1][1]].shape)
                read_scores_dict[read_part[0]][0].append(read[1]['trans'][read_part[1][0]:read_part[1][1]])
                read_scores_dict[read_part[0]][1].append(read[1]['init'][read_part[1][0]:read_part[1][1]])
        #print(read[1]['trans']) #i[1] trans-init

    read_scores_dict2={}
    for i in read_scores_dict.keys():
        t=[]
        t.append(torch.cat(read_scores_dict[i][0],dim=0))
        t.append(torch.cat(read_scores_dict[i][1],dim=0))
        read_scores_dict2[i]=t

    for i in read_scores_dict2.values():
        print(i[1].shape)'''





    '''results = thread_iter(
        (read, stitch_results(scores, end - start, chunksize, overlap, model.stride, reverse))
        for ((read, start, end), scores) in unbatchify(scores)
    )
    '''
    

    return thread_iter(
        (read_key,compute_scores2(model,read_key,read_batches_dict[read_key], reverse=reverse)) for read_key in read_batches_dict.keys()
    )




def main():
    #print(datetime.datetime.now())
    #sys.stderr.write("start \n" )
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=[
        'bonito',
        'bonito-ctc',
        'urlstm5',
        'mymodel',
        'bolstm5',
        'urmat',
        'urlstm5taa',
        'bonitoSEnet',
        'CSNetCNN',
        'deBonitoCnn',
        'RGCNN', 
        'BonitoLSTM5TAAModel',
        'ultimate',
        'ultimate2',
        'ultimate4',
        'TAAnano-364C-32T-3256L',
        'CBAMnano-532C-3256L',
        'TAAnano-332C-1256T-3256L',
        'TAAnano-464T-3256L' ,
        'CSnano-c4-c16-c384-l3',
        'CSnano-c4-c16-c324-Taa-l3',
        'CSnano-c64-c324-l3',
        'CSnano-c4-c16-c324-Taa-l3',
        'CSnano-c4-c16-c324-TaaRes-l3',
        'CSnano-c4-c16-c324-TaaRes-gate-l3',
        'TaaRes-gate-CSnano-c4-c16-c324-l3',
        'CSnano-c4-c16-c324-l3-utrf',
        'CSnano-c4-c16-c324-TaaRes10-gate-l5',
        'Rubicall',
        'Rubicall-base',
        'Rubicall-final',
        'SACall',
        'causalLstm'
    ], required = True)
    parser.add_argument("--fast5-dir", type=str, required = True)
    parser.add_argument("--checkpoint", type=str, help='checkpoint file to load model weights', required = True)
    parser.add_argument("--output-file", type=str, help='output fastq file', required = True)
    parser.add_argument("--chunk-size", type=int, default = 2000)
    parser.add_argument("--window-overlap", type=int, default = 400)
    parser.add_argument("--batch-size", type=int, default = 64)
    parser.add_argument("--beam-size", type=int, default = 1)
    parser.add_argument("--beam-threshold", type=float, default = 0.1)
    parser.add_argument("--model-stride", type=int, default = None)
    parser.add_argument("--recursive", action="store_true", default=False)
    parser.add_argument("--reference", default=False)


    quant_parser = parser.add_mutually_exclusive_group(required=False)
    quant_parser.add_argument("--quantize", dest="quantize", action="store_true")
    quant_parser.add_argument("--no-quantize", dest="quantize", action="store_false")
    parser.set_defaults(quantize=None)
    parser.add_argument("--read-ids")
    
    args = parser.parse_args()

    reader = Reader(args.fast5_dir, args.recursive)
    #sys.stderr.write("> reading %s\n" % reader.fmt)

    fmt = biofmt(aligned=args.reference is not None)
    #print(fmt)
    #print(sys.stdout)
    #sys.stderr.write(f"> outputting {fmt.aligned} {fmt.name}\n")

    

    #load_model
    checkpoint_file = args.checkpoint
    output_file = args.output_file
    

    model = load_model(
            checkpoint_file,
            device,
            weights=args.model,
            chunksize=args.chunk_size,
            overlap=args.window_overlap,
            batchsize=args.beam_size,
            quantize=args.quantize,
            use_koi=True,
    )  
    #sys.stderr.write(f"> loading model {args.checkpoint}\n")


    reads = reader.get_reads(
        args.fast5_dir, n_proc=8, recursive=False,
        read_ids=column_to_set(None), skip=False,
        do_trim= True,
        scaling_strategy=None,
        norm_params=(None),
        cancel=process_cancel()
    )

    results = basecall(
        model, reads, reverse=False, rna=None,
        batchsize=args.batch_size,
        chunksize=args.chunk_size,
        overlap=args.window_overlap
    )
    #print(results)

    #print("basecall_end")
    #print(datetime.datetime.now())

    writer_kwargs = {'aligner': None,
                     'group_key': "test",
                     'ref_fn': None,
                     'groups':[],
                     'min_qscore': 0}
    
    num_reads = None
    ResultsWriter = Writer
        
    writer = ResultsWriter(
        fmt.mode, tqdm(results, desc="> calling", unit=" reads", leave=False,
                       total=num_reads, smoothing=0, ascii=True, ncols=100),output=output_file,
        **writer_kwargs)

    t0 = perf_counter()
    writer.start()
    writer.join()
    duration = perf_counter() - t0
    num_samples = sum(num_samples for read_id, num_samples in writer.log)

    sys.stderr.write("> completed reads: %s\n" % len(writer.log))
    sys.stderr.write("> duration: %s\n" % timedelta(seconds=np.round(duration)))
    sys.stderr.write("> samples per second %.1E\n" % (num_samples / duration))
    sys.stderr.write("> done\n")

  


main()