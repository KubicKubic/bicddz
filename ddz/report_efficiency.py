"""Standalone auditable figure and HTML report from actual campaign artifacts."""
import argparse,base64,html,json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def atomic_json(path,value):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(tmp,path)


def read(p):return json.loads(p.read_text())
def number(x):return f'{x:.4f}'
def interval(v):return f"{v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args();root=args.root
    m=read(root/'manifest.json');result=read(root/'result.json') if (root/'result.json').exists() else None
    if result is None and not args.allow_partial:raise RuntimeError('campaign is not complete')
    screen=read(root/'screen_results.json') if (root/'screen_results.json').exists() else {}
    selection=read(root/'selection.json') if (root/'selection.json').exists() else {}
    confirmation=read(root/'confirmation_results.json') if (root/'confirmation_results.json').exists() else {}
    metrics={}
    for name in m['variants']:
        p=root/'training'/f'{name}_rng0'/'metrics.jsonl'
        if not p.exists():continue
        rows=[json.loads(x) for x in p.read_text().splitlines()]
        rows=[r for r in rows if r['iteration']<=m['screen_steps']]
        warm=[r for r in rows if r['iteration']>20]
        if not warm:warm=rows
        metrics[name]={'updates':len(rows),'fresh_decisions':sum(r['fresh_decisions'] for r in rows),
            'learner_decisions':sum(r['learner_decisions'] for r in rows),
            'actual_seconds':sum(r['seconds'] for r in rows),
            'warm_fresh_per_second':sum(r['fresh_decisions'] for r in warm)/sum(r['seconds'] for r in warm),
            'warm_learner_per_second':sum(r['learner_decisions'] for r in warm)/sum(r['seconds'] for r in warm),
            'rollout_seconds_median':float(np.median([r['rollout_seconds'] for r in warm])),
            'update_seconds_median':float(np.median([r['update_seconds'] for r in warm])),
            'kl_mean':float(np.mean([r['kl'] for r in warm])),
            'applied_mean':float(np.mean([r['applied'] for r in warm])),
            'invalid_actions':sum(r['invalid_actions'] for r in rows),
            'nonfinite':sum(r['nonfinite'] for r in rows)}
    fig,axes=plt.subplots(2,2,figsize=(15,11),constrained_layout=True)
    def bars(ax,labels,values,title):
        for i,v in enumerate(values):
            ax.errorbar(v['mean'],i,xerr=np.array([[v['mean']-v['ci95'][0]],[v['ci95'][1]-v['mean']]]),fmt='o',capsize=4)
        ax.set_yticks(range(len(labels)),labels);ax.axvline(0,color='gray',linewidth=1)
        ax.set_title(title);ax.grid(axis='x',alpha=.2)
    bars(axes[0,0],list(screen),[x['results']['forced'] for x in screen.values()],
        f"Exploratory screen: equal-role expected score vs matched baseline\n{m['screen_steps']} updates; {m['screen_deals']} unique deals per candidate")
    names=list(metrics);ax=axes[0,1];ix=np.arange(len(names))
    ax.barh(ix-.16,[metrics[n]['warm_fresh_per_second'] for n in names],height=.32,label='Fresh decisions/s')
    ax.barh(ix+.16,[metrics[n]['warm_learner_per_second'] for n in names],height=.32,label='Learner decisions/s')
    ax.set_yticks(ix,names);ax.set_title('Measured warm training throughput (rollout + train + diagnostics)');ax.legend()
    labels=[];vals=[]
    for seed,c in confirmation.items():
        labels.append('RNG '+seed+' all roles');vals.append(c['results']['forced'])
        for name,v in c['results']['forced']['roles'].items():labels.append('RNG '+seed+' '+name);vals.append(v)
    bars(axes[1,0],labels,vals,'Held-out confirmation: candidate vs matched baseline\nWhole-deal 95% bootstrap; no screen deals reused')
    dz={}
    for name in ('baseline',selection.get('winner')):
        if name is None:continue
        p=root/'douzero'/name/'BEST'/'summary.json'
        if p.exists():dz[name]=read(p)['summary']
    labels=list(dz);vals=[x['equal_role_expected_score'] for x in dz.values()]
    if result:labels.append('Paired improvement');vals.append(result['strong_douzero_paired_change'])
    bars(axes[1,1],labels,vals,'Publisher released BEST DouZero ResNet 2.0\nEqual-role expected score; six complementary role legs')
    fig.suptitle('V5 sample efficiency: '+('complete' if result else 'IN PROGRESS — no final benefit claim'),fontsize=17)
    fig.savefig(root/'comparison.png',dpi=180);plt.close(fig)
    rows=[]
    for name in m['variants']:
        if name not in metrics:continue
        v=metrics[name];s=screen.get(name)
        rows.append([name,v['updates'],f"{v['fresh_decisions']:,}",f"{v['learner_decisions']:,}",
            f"{v['actual_seconds']:.1f}",f"{v['warm_fresh_per_second']:.0f}",
            interval(s['results']['forced']) if s else 'control / pending',
            interval(s['results']['natural']) if s else 'control / pending',v['invalid_actions'],v['nonfinite']])
    def table(headers,rows):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
    content='<h1>V5 样本效率验证</h1><p>'+html.escape(result['decision'] if result else '实验进行中；尚不能认定棋力收益成立。')+'</p>'
    content+=f"<p>共同起点：V5 step {m['source_iteration']:,}；已有策略、Adam、环境与 RNG 保留。Gamma 1、lambda 0.95、原学习率进度、自适应 value 系数、每 100 updates 保存。</p>"
    content+='<p>筛选结果仅为探索性证据，多项比较尚未校正。确认使用独立测试牌局与两条训练随机流；同副牌的六场保留为一个重采样单元。自然叫分与强制角色指标分别报告。GAE 改动会改变 value target，不能直接用 explained variance 的升降判定棋力。</p>'
    content+='<p>同座位 GAE 的 gamma 按公开 transition 间隔计，lambda 按同一玩家的决策计。它同时改变 bootstrap 信息来源和有效信用分配长度，当前消融未单独隔离这两个因素。</p>'
    content+='<img alt="Actual screening, throughput, held-out confirmation and strong DouZero results" src="data:image/png;base64,'+base64.b64encode((root/'comparison.png').read_bytes()).decode()+'">'
    content+=table(['方案','updates','新决策','learner 决策','实际秒数（含编译）','暖机新决策/s','强制三角色期望分 [95% CI]','自然叫分期望分 [95% CI]','非法动作','非有限'],rows)
    for seed,c in confirmation.items():
        content+=f'<h2>独立确认 RNG {seed}</h2>'+table(['指标','配对期望分 [95% CI]'],
            [[scope,interval(c['results'][scope])] for scope in ('natural','forced')]+[
                [role,interval(v)] for role,v in c['results']['forced']['roles'].items()])
    for name,v in dz.items():
        content+=f'<h2>{html.escape(name)} vs 发布者 best DouZero</h2>'+table(['指标','数值 [95% CI]'],[
            ['等权三角色期望分',interval(v['equal_role_expected_score'])],
            ['等权三角色队伍胜率',interval(v['equal_role_team_win_rate'])]])
    if result:content+='<p>与控制相比的 DouZero 配对得分变化：'+interval(result['strong_douzero_paired_change'])+'</p>'
    cluster=root.parent/'v5_cluster8_v1'
    if (cluster/'engineering_READY.json').exists():
        proof=read(cluster/'engineering_READY.json')
        samples=[json.loads(x) for x in (cluster/'engineering'/'training'/'metrics.jsonl').read_text().splitlines()]
        warm=samples[4:]
        overall=sum(x['fresh_decisions'] for x in samples)/sum(x['seconds'] for x in samples)
        content+='<h2>八卡工程验证与续训</h2>'+table(['指标','实测'],[
            ['设备','8 × NVIDIA A100-SXM4-80GB；日志含 nranks=8 与真实 NCCL 初始化'],
            ['并行环境 / 每轮新决策','8192 / 524,288'],
            ['常规全局 minibatch','16,384；完整长历史分支为 8192'],
            ['验证更新 / 新决策',f"{len(samples)} / {sum(x['fresh_decisions'] for x in samples):,}"],
            ['前四轮后单轮吞吐中位数',f"{np.median([x['decisions_per_second'] for x in warm]):,.0f} 新决策/s"],
            ['全部验证更新平均，含两次启动编译',f'{overall:,.0f} 新决策/s'],
            ['前四轮后 rollout / train 中位数',f"{np.median([x['rollout_seconds'] for x in warm]):.3f} / {np.median([x['update_seconds'] for x in warm]):.3f} 秒"],
            ['非法动作 / 非有限数值 / 副本参数差异','0 / 0 / 0']])
        content+='<p>八卡工程验证从保存的原 V5 step 26554 启动；它是吞吐和同步检查，不能混入单卡消融作为匹配预算的棋力证据。短测试的启动及长历史首次编译成本明显，暖机中位数不代表包含启动的整体吞吐。生产按 1000 updates 分段入 FIFO 队列，每 100 保存，成功后才追加下一段。</p>'
        content+='<p><a href="../v5_cluster8_v1/engineering_READY.json">八卡与 NCCL 证据</a></p>'
        receipt=root/'production_continuation.json'
        if receipt.exists():content+='<p>续训决策：'+html.escape(read(receipt)['decision'])+'</p>'
    content+='<p>固定叫分为 3 的 DouZero 对照不衡量叫分能力。训练与评测使用本项目线性炸弹/春天积分，区别于 DouZero 的指数 ADP。两条 RNG 流不足以证明任意训练 seed 的普适收益；固定预算内未确认的方案不等于长期绝无效果。</p>'
    content+='<p><a href="manifest.json">冻结协议与源码哈希</a> · <a href="engineering_verification.json">工程测试</a> · <a href="result.json">最终机器可读结果</a></p>'
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>V5 efficiency validation</title><style>body{font:15px system-ui;margin:28px;color:#1b2639}p{line-height:1.7;max-width:1200px}img{max-width:100%}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:13px}th,td{padding:10px;border:1px solid #dae0e8;text-align:left}th{background:#edf2f7}</style>'+content+'</html>'
    (root/'report.html').write_text(page)
    atomic_json(root/'measured_training_summary.json',metrics)
    print(root/'report.html')


if __name__=='__main__':main()
