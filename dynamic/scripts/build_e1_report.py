#!/usr/bin/env python3
import sys,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.feature_io import arguments,configuration,write_json,digest,read_jsonl
from src.plotting import build_figures,read_csv


def main():
    c=configuration(arguments('Build measured E1 report and five figure groups').parse_args())
    if 'models' in c:
        from src.comparison_report import build_model_report
        build_model_report(c);return
    out=Path(c['output_root'])
    da=json.loads((out/'audit/dataset_audit.json').read_text());ma=json.loads((out/'audit/model_audit.json').read_text());gate=json.loads((out/'audit/extraction_full_validation.json').read_text());fidelity=json.loads((out/'audit/forward_fidelity.json').read_text())
    assert gate['passed'] and fidelity['passed']
    groups=read_csv(out/'metrics/group_metrics.csv');layers=read_csv(out/'metrics/layer_metrics.csv');gs=sorted(da['core'])
    def val(model,regime,rep,l,metric,phase=None,g=None):
        phase=phase or ('norm' if model=='dinov3' else ('fused' if rep=='dense' else 'post'))
        rr=groups if g else layers
        return next(r['median'] for r in rr if r['model']==model and r['regime']==regime and r['representation']==rep and r['layer']==l and r['metric']==metric and r['phase']==phase and (g is None or r['group']==g))
    def v(reg,rep,l,m,phase=None,g=None):return val('vggt_omega',reg,rep,l,m,phase,g)
    def contrasts(rep,a,b):return [v(a,rep,23,'D_cause',g=g)-v(b,rep,23,'D_cause',g=g) for g in gs]
    def positives(values):return int(np.sum(np.array(values)>0))
    figures=build_figures(c)
    endpoint=[]
    for model,regimes in [('dinov3',['Single']),('vggt_omega',['Single','Pair','Full'])]:
        for reg in regimes:
            for rep in (['patch','pool'] if model=='dinov3' else ['patch','pool','camera','register','dense']):
                ms=['D_cause','C_norm','S_e','S_o'] if rep=='dense' else ['TC_e','TC_o','scale_e','scale_o','cos_signed','cos_abs','S_e','S_o','D_cause','C_norm']
                endpoint.append({'model':model,'regime':reg,'representation':rep,**{m:val(model,reg,rep,23,m) for m in ms}})
    routing=[]
    for reg in ['Pair','Full']:
        for l in c['vggt_omega']['register_only_layers']:
            for rep in ['patch','camera','register']:
                ds=[v(reg,rep,l,'D_cause','post',g)-v(reg,rep,l,'D_cause','pre',g) for g in gs]
                routing.append({'regime':reg,'layer':l,'representation':rep,'post_minus_pre_by_group':ds,'median':float(np.median(ds)),'positive_groups':positives(ds)})
    statements=[
      'Tangent consistency is low/mixed and norm scaling is far from 1; these are finite intervention differences, not established stable local tangent directions.',
      'Early patch ego/object directions are nearly antiparallel. Final patch absolute cosine remains high; selective ego invariance or persistent orthogonalization is not established.',
      'Pair/Full patch same-r separation rises in late layers, peaks before the final layer and then falls; it is not monotonic.',
      f'Final patch Pair > Single in {positives(contrasts("patch","Pair","Single"))}/4 groups; Full > Single in {positives(contrasts("patch","Full","Single"))}/4; Full > Pair in {positives(contrasts("patch","Full","Pair"))}/4.',
      'DINOv3 already has nonzero same-r separation, so static global visual context accounts for part of the distinction.',
      'Camera/register separation is already high at the earliest sampled layer; no clean onset or causal routing is established.',
      f'Dense fused Pair > Single in {positives(contrasts("dense","Pair","Single"))}/4 groups and Full > Single in {positives(contrasts("dense","Full","Single"))}/4; dense Full > Pair in {positives(contrasts("dense","Full","Pair"))}/4.',
      'Dense separation is substantially larger than patch separation in multi-frame regimes; spatial resolution and decoder processing both remain explanations.'
    ]
    summary={'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),'data':{k:da[k] for k in ['dataset_root','scenes','anchors','groups','sequences','frames','delta_m','resolution','core']},'models':{name:{**{k:ma[name][k] for k in ['repo','checkpoint','checkpoint_sha256','variant','git_sha']},'runtime':json.loads((out/'audit'/f'{name}_runtime.json').read_text())} for name in ['dinov3','vggt_omega']},'correctness':{'full':gate,'forward_fidelity':fidelity,'same_r_uv_error_px':da['live_max_same_r_final_uv_error_px']},'endpoint_group_medians':endpoint,'register_only_routing':routing,'findings':statements,'figures':figures,'statistical_unit':'four groups nested in two scenes; no p-values','recommendations':{'more_scenes':'Worth a controlled modest expansion after scale calibration; current direction is pilot evidence only','delta':'Do not simply increase. First compare smaller and larger finite-difference scales and inspect rendering/feature numerical floors; TC currently fails a strong local-linearity interpretation','frames16':'Not currently prioritized: Full does not improve over Pair consistently','register_swap':'Exploratory small intervention on Pair/Full only, especially blocks 2 and 20, with matched-r donor, static/identity controls and downstream patch/dense readouts; not yet a validated routing claim'}}
    report=[]
    def add(s=''):report.append(s)
    def table(headers,rr):
        cell=lambda x:str(x).replace('|','\\|')
        add('| '+' | '.join(cell(x) for x in headers)+' |');add('| '+' | '.join(['---']*len(headers))+' |')
        for row in rr:add('| '+' | '.join(cell(x) for x in row)+' |')
        add()
    add('# E1 pilot report\n')
    add('已实际完成 audit → 40-shard smoke → 400-shard full extraction → metrics → figures。科学结论：**多帧 context 改变并在部分深层增强 same-r separation，但当前结果不支持稳定 tangent orthogonalization，也不支持 Full 一定优于 Pair。**\n')
    add('## 数据与模型\n')
    add(f'数据根目录：`{da["dataset_root"]}`。{da["scenes"]} scenes，{da["anchors"]} anchors，{da["groups"]} groups，{da["sequences"]} sequences，{da["frames"]} frames；每序列 8 帧，δ={da["delta_m"]} m，512×512。e/o 为 world-X 位移等级，实际振幅=等级×δ，r=o−e。\n')
    table(['Group','Scene / anchor / camera','共同 core points','Fallback'],[[f'G{i+1}',g,da['core'][g]['count'],False] for i,g in enumerate(gs)])
    add('每序列有 192 个 canonical surface tracks，含 local/world/camera XYZ、UV、front/in-image/visible、axial Z、ray range、depth-buffer range/error。所有点都有逐帧 visibility 标记；仅通过所有 25 个 final-frame 条件的点进入 primary。四组来自两个场景且均继承 train split；没有训练 probe，也没有独立测试集泛化结论。\n')
    for name in ['dinov3','vggt_omega']:
        runtime=summary['models'][name]['runtime']
        add(f'- **{ma[name]["variant"]}**：source `{ma[name]["repo"]}`；checkpoint `{ma[name]["checkpoint"]}`；SHA256 `{ma[name]["checkpoint_sha256"]}`。D={runtime["embed_dim"]}，patch={runtime["patch_size"]}，depth={runtime["depth"]}，layers={runtime["layers"]}。')
    add('\nDINOv3 本地 checkpoint 是 Hugging Face 格式，实际使用现有 Transformers DINOv3ViTModel 离线加载；没有转换/下载/复制权重。Omega 使用实际本地仓库。两者空间输入均为原始 512×512，RGB/255 后按 ImageNet mean/std 归一化；Omega 在 aggregator 内执行归一化。eval + inference_mode，attention bf16，DenseHead float32；compact feature 保存 float32，避免额外 FP16 存储量化污染小幅 delta。\n')
    add('## 特征与 correctness\n')
    add('I7 始终是 target：Single=[I7]，Pair=[I0,I7]，Full=[I0,…,I7]。每个 regime 使用自己的 static (0,0) I7 特征作 reference。Single 的 camera/register 是 first-frame role；Pair/Full 是 subsequent role，不能把 Single→Pair 特殊 token 差值归为纯增加一帧效应。\n')
    add('Omega cached `[1,S,1041,2048]` 按末维拆为 pre/post 各 1024；camera=0，registers=1:17，patch=17:1041。保存 patch `[P,1024]`、pool `[1,1024]`、camera `[1,1024]`、完整 register `[16,1024]`。DINO hook 后 shared LayerNorm，`[1,1029,1024]` 的 patches 从 index 5 开始。Dense `dense_head.proj` 的 pre-hook 输入 `[1,256,128,128]`，保存同点 `[P,256]`。没有保存完整 patch tensor。\n')
    add('UV 使用 Blender 图像边缘坐标，模型输入变换为 identity；patch-grid=(UV/16−0.5)，Dense-grid=(UV/4−0.5)，row-major reshape，align_corners=False 双线性采样。插值是 latent lattice readout，不能当成真正 pixel-level feature。边界距离采用插值 EDT−0.5 px，≥8 px；primary 是同物理点，≥90% occupancy pool 是独立的 entity robustness。\n')
    disp=da['displacements'];counts=[r['clean_patches'] for r in da['clean_patch_counts']]
    table(['Check','Measured result'],[
      ['全量完整性',f'{gate["shards_checked"]}/{gate["expected_shards"]} shards; exact point IDs and input indices'],
      ['NaN / Inf',gate['nonfinite']],['register-only raw patch max |pre−post|',gate['register_only_patch_max_error']],
      ['所有同 r 配对 final UV 最大差 (px)',f'{da["live_max_same_r_final_uv_error_px"]:.3g}'],
      ['track reprojection 最大坐标差 (px)',f'{da["live_max_track_reprojection_error_px"]:.3g}'],
      ['Dense endpoint-only vs full head max abs',f'{fidelity["endpoint_only_vs_all_frames_dense_max_abs"]:.3g}'],
      ['重复 forward vs 已保存 Full static shard 最大差',fidelity['saved_shard_deterministic_repeat_max_abs']],
      ['clean pooling patch count range',f'{min(counts)}–{max(counts)}'],
      ['feature storage','float32; ~1.1 GB compact shards']])
    for level in [1,2]:
        vv=[r['median_px'] for r in disp if max(abs(r['e']),abs(r['o']))==level]
        add(f'|level|={level} 的 pure ego/object endpoint displacement：各条件组内中位数范围 {min(vv):.3f}–{max(vv):.3f} px，即 {min(vv)/16:.3f}–{max(vv)/16:.3f} patch。')
    add('\n## 初步 E1 结果（group medians，再跨组 median）\n')
    add('### 1–3. Tangent consistency、反向同轴与 sensitivity\n')
    table(['Model / regime','Layer','TC_e','TC_o','signed cos','|cos|','S_e','S_o','scale_e','scale_o'],[[model+' '+reg,l]+[f'{val(model,reg,"patch",l,m):.3f}' for m in ['TC_e','TC_o','cos_signed','cos_abs','S_e','S_o','scale_e','scale_o']] for model,reg,l in [('dinov3','Single',5),('dinov3','Single',23)]+[('vggt_omega',reg,l) for reg in ['Single','Pair','Full'] for l in [0,23]]])
    add('TC 偏低/混合，inner/outer norm ratio 明显不接近 1，当前尺度不能视为已验证的稳定 local tangent regime。早期 signed cosine 接近 −1，符合 shared relative-motion axis；最终 patch |cos| 仍高。中间层变化并非单调，不能挑某一层宣称持续 orthogonalization。Omega 两种 sensitivity 都随网络尺度显著增大，未观察到最终 ego sensitivity 单独消失；跨模型 raw norm 的尺度不同，不宜直接比较信息量。\n')
    add('### 4–6. Same-r causal distance、input regime 与 DINO baseline\n')
    table(['Omega layer','Single pre','Single post','Pair pre','Pair post','Full pre','Full post'],[[l]+[f'{v(reg,"patch",l,"D_cause",phase):.3f}' for reg in ['Single','Pair','Full'] for phase in ['pre','post']] for l in c['vggt_omega']['layers']])
    table(['DINO layer','D_cause'],[[l,f'{val("dinov3","Single","patch",l,"D_cause"):.3f}'] for l in c['dinov3']['layers']])
    add('每点先对 20 个 nonzero-r matched pairs 的 D 取均值，再组内取 median；不是 absolute clustering。Pair/Full 在深层出现更强分离并在最终层回落。DINO 本身已有分离，故不能说只有 reconstruction 模型能识别 causal source。归一化距离也可反映 global appearance/context，不是物理因果识别准确率。\n')
    table(['Group','DINO final','Omega Single final','Pair final','Full final'],[[f'G{i+1}',f'{val("dinov3","Single","patch",23,"D_cause",g=g):.3f}']+[f'{v(reg,"patch",23,"D_cause",g=g):.3f}' for reg in ['Single','Pair','Full']] for i,g in enumerate(gs)])
    add(f'最终 patch Pair>Single：{positives(contrasts("patch","Pair","Single"))}/4；Full>Single：{positives(contrasts("patch","Full","Single"))}/4；Full>Pair：{positives(contrasts("patch","Full","Pair"))}/4。Pair/Full layer23 比 layer0 的 D 增大均为 4/4，但不能称为全层单调改善。\n')
    add('### 7. Camera/register routing\n')
    table(['Regime','Representation','L0 D','L23 D','L23 |cos|'],[[reg,rep,f'{v(reg,rep,0,"D_cause"):.3f}',f'{v(reg,rep,23,"D_cause"):.3f}',f'{v(reg,rep,23,"cos_abs"):.3f}'] for reg in ['Pair','Full'] for rep in ['patch','camera','register']])
    table(['Regime','Register-only block','median register ΔD post−pre','正向组数'],[[r['regime'],r['layer'],f'{r["median"]:+.4f}',f'{r["positive_groups"]}/4'] for r in routing if r['representation']=='register'])
    add('Camera/register 在最早采样层就有较高 separation，而 patch 较低；不能据此确定信息最初出现的时间。Block 2 register ΔD 在四组同向增加，block 20 在四组同向降低，而同层 patch 严格不变；并非每次 inter-frame exchange 都增加 separation。后续 block 的 patch 改变仅为 observational mechanism evidence。稀疏层采样不能证明 register→patch causal routing。\n')
    add('### 8–9. Dense/pool robustness 与跨组一致性\n')
    table(['Representation','Regime','D_cause','C_norm','S_e','S_o'],[[rep,reg]+[f'{v(reg,rep,23,m):.3f}' for m in ['D_cause','C_norm','S_e','S_o']] for rep in ['patch','pool','dense'] for reg in ['Single','Pair','Full']])
    add(f'Dense Pair>Single：{positives(contrasts("dense","Pair","Single"))}/4；Full>Single：{positives(contrasts("dense","Full","Single"))}/4；Full>Pair：{positives(contrasts("dense","Full","Pair"))}/4。Pool Pair/Full 对 Single 也同向提升，但其物理点集合语义不同。Dense multi-frame D 明显高于 primary patch，方向支持 context distinction，同时提示 spatial resolution 或 decoder 处理影响信号；不能宣称已排除 resolution artifact。\n')
    add('## 下一步建议\n')
    add('1. 值得做适度 scene 扩展，以验证 Pair/Full 相对 Single 的跨组方向，但先校准 perturbation scale。四组共享两场景，不是四个完全独立 scene。\n2. 不建议直接加大 δ。先在保持场景配对的前提下做更小/更大步长对照、检查渲染变化和数值底噪，再判断哪些尺度可近似局部；现在 TC 和 norm scaling 不支持稳定线性。\n3. 暂不优先增加到 16 帧；当前 Full 相比 Pair 没有稳定优势。\n4. Register activation swap 值得小规模探索，但只是 hypothesis test：优先 Pair/Full 的 block 2（分离小幅增加）和 20（明显下降）作为对照，配 matched-r donor、static/identity swap control，观察后续 patch/dense 指标。当前证据不足以称为已验证的 routing。\n')
    add('## 复现与文件\n')
    add('```bash\nbash /home/3dsm/Desktop/JSWprobe/dynamic/scripts/run_e1_pipeline.sh\n```\n')
    add('Config: `dynamic/configs/e1_pilot.yaml`；代码模块：dataset、spatial_sampling、两模型 adapter、feature_io、metrics、plotting；脚本：audit、extract、validate、forward_fidelity、analysis、report、pipeline。精确定义见 `dynamic/docs/feature_definitions.md` 与 `e1_protocol.md`。\n')
    add('CSV：`point_metrics.csv`、`group_metrics.csv`、`layer_metrics.csv`、`matched_relative_metrics.csv`、`register_metrics.csv`、`dense_aux_metrics.csv`。所有图都有 PNG/PDF；淡线或点表示单组，没有显著性检验。\n')
    for f in figures:add(f'![{f}](../figures/{f})\n')
    dest=out/'report';dest.mkdir(exist_ok=True);write_json(dest/'E1_SUMMARY.json',summary);(dest/'E1_REPORT.md').write_text('\n'.join(report))
    write_json(dest/'artifact_manifest.json',{'created_utc':summary['created_utc'],'artifacts':{str(p.relative_to(out)):digest(p) for folder in ['metrics','figures','report'] for p in sorted((out/folder).glob('*')) if p.is_file() and p.name!='artifact_manifest.json'},'implementation_sha256':{str(p.relative_to(Path(__file__).resolve().parents[1])):digest(p) for p in sorted(Path(__file__).resolve().parents[1].rglob('*')) if p.suffix in ['.py','.yaml','.sh','.md'] and 'outputs' not in p.parts}})
    print('Report:',dest/'E1_REPORT.md','Figures:',len(figures))
if __name__=='__main__':main()
