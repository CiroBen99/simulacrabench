"""Native Kumo context cache, canonical classes and bounded query chunks."""
import time
import numpy as np


def align(frame,width,observed):
    classes=[int(c) for c in frame.columns]
    if set(classes)!=set(observed) or len(classes)!=len(set(classes)) or not set(classes).issubset(range(width)):
        raise ValueError('Cached class alignment failed')
    p=np.zeros((len(frame),width))
    p[:,classes]=frame.to_numpy()
    if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-6):
        raise ValueError('Cached probabilities invalid')
    return p


def forward_bridge(device):
    """Keep recipe preprocessing/RNG on CPU; transfer only native forward inputs."""
    from sdm.models.callback import Callback
    class Bridge(Callback):
        def on_context_preprocessing_end(self,model,x,y,related):
            if related is not None:raise ValueError('Only ordinary tabular context supported')
            return x.to(device),y.to(device),None
        def on_query_preprocessing_end(self,model,x,related):
            if related is not None:raise ValueError('Only ordinary tabular query supported')
            return x.to(device),None
    return Bridge()


def cached_probabilities(engine,x,y,z,width,chunk_size=256,deadline=None,estimators=1):
    import pandas as pd
    import torch
    if chunk_size<1 or list(x.columns)!=list(z.columns) or len(x)!=len(y) or not len(z):
        raise ValueError('Invalid context/query contract')
    sdm=engine.sdm
    table=pd.concat([x,z],ignore_index=True)
    table=sdm.TableTensor.from_pandas(table,stypes={k:'categorical' for k in table.columns},device='cpu')
    labels=sdm.TableTensor.from_pandas(pd.DataFrame({'target':y}),stypes={'target':'categorical'},device='cpu')
    result=[]
    forward_device=getattr(engine,'forward_device',None)
    callbacks=[forward_bridge(forward_device)] if forward_device else []
    call_kwargs={'callbacks':callbacks} if callbacks else {}
    try:
        if deadline is not None and time.perf_counter()>deadline:
            raise TimeoutError('Context admission deadline reached')
        fit_options={'estimator_batch_size':1} if estimators>1 else {}
        engine.model.fit(table[:len(x)],labels,num_estimators=estimators,**fit_options,generator=torch.Generator().manual_seed(0),**call_kwargs)
        for offset in range(0,len(z),chunk_size):
            if deadline is not None and time.perf_counter()>deadline:
                raise TimeoutError('Cached query deadline reached; keep baseline for entire target')
            pred=engine.model.predict(table[len(x)+offset:len(x)+min(offset+chunk_size,len(z))],**call_kwargs)
            result.append(align(pred.to_pandas(),width,y))
    finally:
        engine.model.clear()
    engine.used=True
    return np.concatenate(result)
