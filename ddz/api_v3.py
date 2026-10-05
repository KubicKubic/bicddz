"""Server adapter for exact candidate-set DDZ policy."""
import argparse
import json
import os
from pathlib import Path
import jax
import jax.numpy as j
from flax import serialization

from . import env_v2 as env
from . import candidates_v3 as moves
from .actions import BODIES,physical_cards
from .api_v2 import from_api
from .api import Client,run
from .model_v3 import CandidateSetTransformer


class Policy:
    def __init__(self,config,checkpoint,memory_limit=88):
        cfg=json.loads(Path(config).read_text())
        self.model=CandidateSetTransformer(**cfg['model'])
        self.params=serialization.msgpack_restore(Path(checkpoint).read_bytes())
        self.memory_limit=memory_limit
        self.forward=jax.jit(lambda obs,ids,mask,memory_length:
            self.model.apply({'params':self.params},obs,ids,mask,
                             memory_length=memory_length),
            static_argnames=('memory_length',))
        state=env.reset(jax.random.PRNGKey(0))
        self._score(state)

    def _score(self,state):
        batch=jax.tree_util.tree_map(lambda x:x[None],state)
        ids,mask,_=moves.batch_legal(batch)
        obs=jax.vmap(env.observe)(batch)
        memory_length=self.memory_limit if int(state.hist_len)<=self.memory_limit else env.HISTORY
        logits,_=self.forward(obs,j.asarray(ids),j.asarray(mask),memory_length)
        return ids[0,int(j.argmax(logits[0]))]

    def decide(self,raw):
        state=from_api(raw)
        cid=int(self._score(state))
        action=int(moves.ACTION[cid])
        version=int(raw['version'])
        if action<0:
            return 'bid',{'version':version,'value':-action-1}
        body=int(moves.BODY[cid])
        if body==0:
            return 'pass',{'version':version}
        cards=physical_cards(raw['hand'],moves.CARDS[cid],raw.get('bottom') or ())
        return 'play',{'version':version,'cards':cards,
                       'choice':BODIES[body].choice}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',required=True)
    ap.add_argument('--token-env',default='DDZ_API_KEY')
    ap.add_argument('--config',required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--memory-limit',type=int,default=88)
    ap.add_argument('--mode',choices=['single','match'],default='single')
    ap.add_argument('--once',action='store_true')
    args=ap.parse_args()
    policy=Policy(args.config,args.checkpoint,args.memory_limit)
    run(Client(args.base,os.environ[args.token_env]),policy,args.mode,args.once)


if __name__=='__main__':main()
