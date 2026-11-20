"""Pure frozen Kumo per recorded target; no learned statistical/model fallback."""
import hashlib
import json
import time
import types
from pathlib import Path
import numpy as np
from .episodic_cached import cached_probabilities
from .schema import cells_for,items_for,options_for,validate_vectors


def contract(schema):
    data={'gated_value':schema.get('gated_value'),'items':[[n,r['class'],r['values'],r.get('gate')] for n,r in schema['items'].items() if r['class'] in ('GIVEN','PREDICT')]}
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


class FrozenKumo:
    def __init__(self,root,config):
        import torch
        import sdm
        device=('cuda' if torch.cuda.is_available() else 'cpu') if config['device']=='auto' else config['device']
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Kumo-only requires CUDA; no silent CPU/model fallback')
        path=Path(root)/'kumo-small.pt';h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=config['checkpoint_sha256']:raise ValueError('Frozen Kumo checkpoint changed')
        class Small(sdm.models.KumoTabular):
            def _load_from_pretrained(inner,size,device):
                if size!='small' or len(inner.models)!=1:raise ValueError('Wrong model recipe')
                model=next(iter(inner.models.values()));model.load_state_dict(torch.load(path,map_location=device,weights_only=True),assign=True)
                return model
        torch.set_num_threads(config['threads']);self.sdm=sdm;self.model=Small(task='classification',size='small',device=device);self.used=False;self.forward_device=device if device!='cpu' else None
        original=self.model.ecoc._draw_codebook
        def cpu_codebook(inner,count,target_device,generator):
            return original(count,torch.device('cpu'),generator).to(target_device)
        self.model.ecoc._draw_codebook=types.MethodType(cpu_codebook,self.model.ecoc)


class KumoOnly:
    def __init__(self,config,root,started=None,engine_factory=None,clock=time.perf_counter):
        if config['method']!='kumo_only' or config['device'] not in ('cuda','cpu','auto'):raise ValueError('Invalid pure Kumo configuration')
        self.config=config;self.root=root;self.clock=clock;self.started=clock() if started is None else started;self.engine_factory=engine_factory or (lambda:FrozenKumo(root,config));self.engine=None;self.phase=None
        self.diagnostics=dict(actual_model_used=False,statistical_fallback=False,neural_gradient_updates=False,instruments={})

    def predict(self,frame,schema):
        targets=items_for(schema,('PREDICT',));given=items_for(schema,('GIVEN',));cells=cells_for(frame,schema)
        if not cells:return []
        missing=frame[targets].isna()
        if (missing.any(axis=1)!=missing.all(axis=1)).any():raise ValueError('Require fully masked query PREDICT')
        visible=~missing.any(axis=1).to_numpy();queries=np.flatnonzero(~visible)
        if not len(queries):return []
        if not visible.any():raise ValueError('No complete Kumo context; no statistical fallback')
        for n in given:
            if not frame[n].isin(options_for(schema,n)).all():raise ValueError('Invalid GIVEN category')
        for n in targets:
            if not frame.loc[visible,n].isin(options_for(schema,n)).all():raise ValueError('Invalid recorded response')
        instrument=self.config['instrument_contracts'].get(contract(schema),'unknown');split=schema.get('split',{});final=(split.get('n_train',-1)+split.get('n_dev',-1),split.get('n_test',-1));shape=(int(visible.sum()),len(queries));phase='final' if shape==final and min(final)>0 else 'development'
        if self.phase is not None and phase!=self.phase:raise ValueError('Mixed competition phases')
        self.phase=phase;deadline=self.started+self.config['phase_total_seconds'][phase]
        cap=self.config['support_caps'].get(instrument,self.config['default_support_cap']);donors=np.flatnonzero(visible)[:cap];x=frame.iloc[donors][given].astype(str);z=frame.iloc[queries][given].astype(str);matrices={};constant=[];native=[]
        for n in targets:
            if self.clock()+self.config['minimum_admission_seconds']>=deadline:raise TimeoutError('Pure Kumo shared budget exhausted; no replacement model')
            levels=options_for(schema,n);y=frame.iloc[donors][n].map({v:i for i,v in enumerate(levels)}).to_numpy(int)
            if len(set(y))==1:
                q=np.zeros((len(queries),len(levels)));q[:,y[0]]=1.;constant.append(n)
            else:
                if self.engine is None:self.engine=self.engine_factory()
                q=cached_probabilities(self.engine,x,y,z,len(levels),chunk_size=self.config['chunk_size'],deadline=deadline,estimators=self.config['estimators']);native.append(n)
                if self.clock()>deadline:raise TimeoutError('Pure Kumo exceeded shared budget')
            matrices[n]=q
        position={int(row):i for i,row in enumerate(queries)};result=[matrices[n][position[row]].tolist() for row,_,n in cells];validate_vectors(result,schema,cells)
        self.diagnostics['actual_model_used']=bool(self.engine is not None and self.engine.used)
        self.diagnostics['instruments'][instrument]=dict(context_people=len(donors),query_people=len(queries),native_targets=native,constant_targets=constant,all_targets=len(targets),phase=phase)
        return result
