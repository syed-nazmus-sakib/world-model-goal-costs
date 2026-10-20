"""Create publication-style figures and genuine simulator animations from saved data.

Usage: python scripts/visuals/make_visual_story.py
Dependencies: numpy scipy matplotlib pillow. Run render_remote_assets.py first.
"""
from pathlib import Path
import json
import textwrap
import html

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import ListedColormap
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/visual_story_2026-10-16"
DATA = OUT / "source_data"
PODS = ROOT / "artifacts/raw_logs/pod_runs"
BLUE, GREEN, RED, PURPLE, GRAY = "#3976ac", "#287b67", "#ba5149", "#7b699e", "#6b7280"
INK, LIGHT = "#202733", "#f4f6f8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.labelcolor": INK, "text.color": INK, "axes.titleweight": "semibold",
    "svg.fonttype": "none", "figure.facecolor": "white", "savefig.facecolor": "white"})
manifest = {"created": "2026-10-16", "scope": "Existing experiments only; no new training or planning", "figures": {}}


def save(fig, name, caption, sources):
    for ext in ["png", "svg"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    manifest["figures"][name] = {"caption": caption, "sources": sources}


def success(path, key="sim_latent_success"):
    files = sorted(path.glob("ep_*.npz"))
    values = []
    for p in files:
        with np.load(p) as z:
            values.append(bool(z[key]))
    return np.array(values)


def foot(fig, text):
    fig.text(0.02, 0.015, text, fontsize=9, color=GRAY, va="bottom")


def pipeline(replay, drawer):
    fig = plt.figure(figsize=(17, 10.7))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, 17), ylim=(0, 10.7)); ax.axis("off")
    ax.text(.55, 10.25, "When the goal image says more than the task", fontsize=24, weight="bold")
    ax.text(.55, 9.8, "Project map: visual planning, controlled failure diagnosis, and a tested idea that does not yet generalize", fontsize=12, color=GRAY)
    def box(x, y, w, h, title, body, color=BLUE, dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=.12", facecolor=color+"16", edgecolor=color, linewidth=1.4, linestyle="--" if dashed else "-"))
        ax.text(x+w/2, y+h-.3, title, ha="center", va="top", fontsize=12, weight="bold", color=color)
        ax.text(x+w/2, y+h/2-.12, body, ha="center", va="center", fontsize=10, linespacing=1.5)
    def arrow(x1,y1,x2,y2,color=INK,dashed=False):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=16,linewidth=1.4,color=color,linestyle="--" if dashed else "-"))
    def pic(frame,x,y,w=1.3,h=1.3):ax.imshow(frame,extent=(x,x+w,y,y+h),zorder=2,aspect="auto")
    ax.text(.55,9.3,"A  THE EXISTING PLANNER",fontsize=12,weight="bold",color=GRAY)
    pic(replay["wm_frames"][0],.6,7.55)
    ax.text(1.25,7.28,"Current scene",ha="center",fontsize=10)
    box(2.45,7.6,2.15,1.25,"Frozen DINOv2","Image → patch features\nNo encoder training")
    box(5.25,7.6,2.7,1.25,"Candidate futures","Released world model\nor exact simulator")
    box(8.65,7.6,2.45,1.25,"Goal cost","Feature distance\n+ α · robot-state distance")
    box(11.75,7.6,1.75,1.25,"CEM search","Sample, score,\nrefine actions")
    pic(replay["sim_latent_frames"][-1],14.2,7.55)
    ax.text(14.85,7.28,"Execute and evaluate",ha="center",fontsize=10)
    for ends in [(1.95,8.2,2.3,8.2),(4.75,8.2,5.1,8.2),(8.1,8.2,8.5,8.2),(11.25,8.2,11.6,8.2),(13.65,8.2,14.08,8.2)]:arrow(*ends)
    pic(replay["goal_frame"],8.95,5.65,1.05,1.05)
    ax.text(10.15,6.17,"One goal image\ncontains object + robot pose",fontsize=10,va="center")
    arrow(9.5,6.8,9.5,7.45)
    ax.plot([.55,16.4],[5.25,5.25],color="#d8dde3",lw=1)
    ax.text(.55,4.9,"B  OUR CONTROLLED TESTS — CHANGE ONE COMPONENT",fontsize=12,weight="bold",color=GRAY)
    box(.7,2.65,4.3,1.75,"Dynamics × cost intervention","Push-T, 96 episodes per condition\nLearned / latent: 64.6%\nExact / latent: 64.6%    Exact / true: 90.6%",GREEN)
    pic(drawer["goal_frames"][0],5.65,2.7,1.45,1.45)
    pic(drawer["goal_frames"][3],7.4,2.7,1.45,1.45)
    ax.text(7.25,4.35,"Same drawer outcome, different arm pose",ha="center",fontsize=11,weight="bold")
    ax.text(7.25,2.3,"Original goal 48/48  →  arm-moved goal 36/48\nExact dynamics, α = 0; 48 paired episodes",ha="center",fontsize=10,linespacing=1.5)
    box(10.05,2.65,5.95,1.75,"Goal sets: useful diagnostic, failed general method","Drawer v3: 17/24 → 22/24 with learned goal set\nBut original goal: 24/24 → 21/24; reach-wall: 23/24 → 0/24\nOracle robot mask + α = 0.1: 24/24 on drawer v3",RED)
    box(1,.65,15,.85,"OPEN RESEARCH QUESTION — NOT A VALIDATED METHOD","How can a goal representation ignore task-irrelevant factors while preserving the robot pose when the task requires it?",PURPLE,True)
    ax.text(.65,.18,"Actual saved simulator images. Aggregate values: JEPA-WM configuration. Goal-set batches n=24; oracle-mask render uses privileged simulator information.",fontsize=9,color=GRAY)
    save(fig,"01_project_overview","The project's existing planner, interventions, established goal-image failure, and unresolved method question.",["docs/07_RESTART_LOG.md","artifacts/raw_logs/pusht_failure_attribution/jepa_wm/ep_002.npz"])


def benchmark_results():
    fig,axs=plt.subplots(1,2,figsize=(13.8,5.1),gridspec_kw={"width_ratios":[2.5,1]})
    x=np.arange(4);w=.24
    series=[("Learned dynamics + latent cost",[64.6,65.6,75,63.5],GRAY),("Exact dynamics + latent cost",[64.6,61.5,83.3,75],BLUE),("Exact dynamics + true-state cost",[90.6,90.6,92.7,92.7],GREEN)]
    for j,(name,val,c) in enumerate(series):
        bars=axs[0].bar(x+(j-1)*w,val,w,label=name,color=c)
        axs[0].bar_label(bars,fmt="%.1f",fontsize=9,padding=3)
    axs[0].set(xticks=x,xticklabels=["Push-T\nJEPA-WM","Push-T\nDINO-WM","Wall\nJEPA-WM","Wall\nDINO-WM"],ylim=(0,107),ylabel="Success rate (%)",xlabel="Benchmark / model configuration",title="Exact dynamics do not remove the goal-cost gap")
    axs[0].legend(frameon=False,loc="upper left",bbox_to_anchor=(0,1.32),ncol=2,fontsize=9)
    b=axs[1].bar([0,1],[2/24*100,20/24*100],color=[BLUE,GREEN],width=.6)
    axs[1].bar_label(b,labels=["2/24","20/24"],padding=4)
    axs[1].set(xticks=[0,1],xticklabels=["Latent\ncost","True-state\ncost"],ylim=(0,107),xlabel="Cost with exact dynamics",ylabel="Success rate (%)",title="Metaworld push")
    for ax in axs:ax.yaxis.grid(True,alpha=.15);ax.set_axisbelow(True)
    foot(fig,"Source: saved attribution summaries. Push-T / Wall: 96 episodes per condition; Metaworld push: 24. One released checkpoint per model; rates rounded.")
    fig.subplots_adjust(bottom=.2,top=.73,wspace=.3)
    save(fig,"02_oracle_attribution_results","Success ladders under controlled dynamics and cost replacements; search budget held fixed.",["docs/01_ASSESSMENT.md","results/jepa_wms"])


def goal_set_diagram(drawer):
    fig=plt.figure(figsize=(16,8.5));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,16),ylim=(0,8.5));ax.axis("off")
    ax.text(.5,8.05,"The tested idea: average model-generated goal representations",fontsize=21,weight="bold")
    ax.text(.5,7.6,"Training-free experiment using frozen released models — not a validated general method",fontsize=12,color=GRAY)
    def box(x,y,w,h,title,body,c=BLUE,dash=False):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.12,rounding_size=.12",facecolor=c+"16",edgecolor=c,lw=1.5,linestyle="--" if dash else "-"))
        ax.text(x+w/2,y+h-.2,title,ha="center",va="top",fontsize=12,color=c,weight="bold")
        ax.text(x+w/2,y+h/2-.1,body,ha="center",va="center",fontsize=11,linespacing=1.5)
    def arr(x1,y1,x2,y2,c=INK):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=16,color=c,lw=1.5))
    ax.imshow(drawer["goal_frames"][3],extent=(.6,2,4.9,6.3),aspect="auto")
    ax.text(1.3,4.55,"Given goal image\n(arm-moved example)",ha="center",fontsize=11)
    box(2.7,5.05,2.0,1.1,"Frozen DINOv2","Goal features z(g)")
    box(5.45,4.8,3,1.6,"Frozen world model","K = 32 sustained directions\nH = 16 planner steps\nPredict final features z₁ … zₖ")
    box(9.2,5,2.35,1.2,"Mean goal target","z̄ = mean[z(g), z₁ … zₖ]",PURPLE)
    box(12.3,5,2.85,1.2,"Planning goal cost","Distance of candidate future\nto z̄ + α · proprio term",PURPLE)
    for p in [(2.1,5.6,2.55,5.6),(4.85,5.6,5.3,5.6),(8.6,5.6,9.05,5.6),(11.7,5.6,12.15,5.6)]:arr(*p)
    ax.plot([3.7,3.7,10.4,10.4],[6.3,6.75,6.75,6.35],color=INK,lw=1.3);arr(10.4,6.75,10.4,6.35)
    ax.text(7.15,6.9,"Include the original goal representation in the mean",ha="center",fontsize=10,color=GRAY)
    box(5.45,2.55,3,1.35,"Reverse-cycle gate tested","Reverse predicted motion;\ntry to detect object changes\nAUC = 0.47 → gate rejected",RED)
    arr(6.95,4.65,6.95,4.05,RED)
    box(9.2,2.55,5.95,1.35,"Measured outcome: partial recovery, failed controls","Drawer v3: 17/24 → 22/24\nOriginal drawer goal: 24/24 → 21/24\nReach-wall: 23/24 → 0/24",RED)
    arr(13.7,4.85,13.7,4.05,RED)
    box(.7,.65,14.45,1.05,"UNRESOLVED: WHICH VARIATIONS ARE SAFE TO IGNORE?","A task-aware mechanism must preserve arm pose when it defines success. Such a mechanism has not been validated.",PURPLE,True)
    ax.text(.65,.18,"Source: goalset.py and saved runs. Magnitude 0.7, K32/H16 result shown; exact simulator dynamics in downstream planning. No decoder or training loss is added.",fontsize=9,color=GRAY)
    save(fig,"10_tested_goal_set_idea","Mechanistic diagram of the frozen-model goal-set experiment, with failed gate and no-harm controls explicitly marked.",["code/jepa-wms/evals/failure_attribution/goalset.py","docs/07_RESTART_LOG.md"])


def goal_figures(task,d):
    frames=d["goal_frames"];obj=d["obj"]
    fig,axs=plt.subplots(1,5,figsize=(14.4,3.9))
    for j,ax in enumerate(axs):
        ax.imshow(frames[j]);ax.axis("off");ax.set_title("Original goal" if j==0 else f"Arm variant {j}",fontsize=11)
    drift=np.linalg.norm(obj[1:]-obj[1],axis=1).max()*1000
    fig.suptitle(f"Metaworld {task}: one task outcome, several goal images",fontsize=16,weight="bold",y=.98)
    foot(fig,f"Actual saved goal renders, episode 002. Variants 1–4 object-position spread ≤ {drift:.2f} mm; variant 0 is the expert goal. These are counterfactual goals, not a time sequence.")
    fig.subplots_adjust(top=.82,bottom=.17,wspace=.06)
    save(fig,f"03_{task}_equivalent_goals",f"Real simulator goal images used in the {task} equivalence test.",[f"source_data/{task}_offline_episode.npz"])


def heatmaps(task,z):
    frames=z["frames"];maps=z["patch_mse"]
    vmax=float(maps.max())
    fig,axs=plt.subplots(2,3,figsize=(11.5,7.8),gridspec_kw={"height_ratios":[1,1]})
    axs[0,0].imshow(frames[2]);axs[0,0].set_title("Fixed scene: variant 2")
    axs[1,0].imshow(np.abs(frames[0].astype(float)-frames[3].astype(float)).mean(-1),cmap="gray",vmin=0,vmax=255)
    axs[1,0].set_title("Pixel difference between goals",fontsize=11)
    for j,k in enumerate([0,3]):
        axs[0,j+1].imshow(frames[k]);axs[0,j+1].set_title(f"Goal {k}: same object outcome")
        axs[1,j+1].imshow(frames[2]);im=axs[1,j+1].imshow(maps[j],cmap="magma",vmin=0,vmax=vmax,extent=(-.5,223.5,223.5,-.5),interpolation="nearest",alpha=.78)
        axs[1,j+1].set_title(f"Patch MSE to goal {k}\nMean = {maps[j].mean():.4f}",fontsize=11)
    for a in axs.flat:a.set_xticks([]);a.set_yticks([])
    fig.suptitle(f"{task.capitalize()}: the same scene receives different DINOv2 costs",fontsize=17,weight="bold",y=.99)
    cax=fig.add_axes([.91,.18,.018,.6]);fig.colorbar(im,cax=cax,label="Feature MSE per patch (shared scale)")
    foot(fig,"Actual DINOv2 ViT-S/14 features; 16×16 patches, mean squared difference over 384 features. ImageNet normalization.\nThis is a cost-contribution map, not attention or causal saliency. Coordinates are image pixels; α term excluded. Episode 002.")
    fig.subplots_adjust(left=.03,right=.88,top=.9,bottom=.14,wspace=.1,hspace=.2)
    save(fig,f"04_{task}_DINO_patch_cost",f"Measured patch-feature discrepancy for one fixed scene and two task-equivalent goals; each task uses its own shared color scale.",[f"source_data/{task}_dino_maps.npz"])


def method_results():
    a=PODS/"podA/drawer_alpha0";b=PODS/"podB/drawer_alpha0"
    conditions=[("Original goal",a/"v0"),("Arm-moved goal v3",a/"v3"),("Oracle mean {2,3,4}",a/"orbit_234"),("Oracle mean {0,…,4} *",a/"orbit_01234"),("Learned set, H16 mag 1.0",b/"wmset_h16k32_v3"),("Learned set, H16 mag 0.7",b/"wmset_h16k32m07_v3"),("Mag 0.7 on ORIGINAL goal",b/"wmset_h16k32m07_v0"),("Oracle mask + α=0.1, v3",PODS/"podA/mask_baselines/drawer_v3_robotfree_a01")]
    vals=np.stack([success(p)[:24] for _,p in conditions]);assert vals.shape==(8,24)
    fig,(ax,bar)=plt.subplots(1,2,figsize=(14,5.8),gridspec_kw={"width_ratios":[3,1.1]})
    ax.imshow(vals,aspect="auto",cmap=ListedColormap(["#f0d5d2",GREEN]),vmin=0,vmax=1,interpolation="nearest")
    ax.set(yticks=np.arange(8),yticklabels=[x[0] for x in conditions],xticks=[0,5,11,17,23],xticklabels=[1,6,12,18,24],xlabel="Matched episode index (first 24 cases)",title="Paired drawer-close outcomes")
    ax.set_xticks(np.arange(-.5,24,1),minor=True);ax.set_yticks(np.arange(-.5,8,1),minor=True);ax.grid(which="minor",color="white",lw=1);ax.tick_params(which="minor",length=0)
    bar.barh(np.arange(8),vals.sum(1),color=[GRAY,RED,BLUE,BLUE,PURPLE,PURPLE,RED,GREEN]);bar.invert_yaxis();bar.set(yticks=[],xlim=(0,27),xticks=[0,12,24],xlabel="Successful episodes / 24",title="Totals")
    for i,v in enumerate(vals.sum(1)):bar.text(v+.4,i,f"{v}/24",va="center",fontsize=10)
    foot(fig,"Green = final-step success; pale red = failure. Exact simulator planning dynamics. α=0 except mask row (α=0.1).\n* Five-goal oracle includes the original goal. Learned-set v3 improvements: paired p=0.0625 (mag 1.0), p=0.125 (mag 0.7); neither <0.05.")
    fig.subplots_adjust(left=.255,right=.96,bottom=.2,top=.87,wspace=.2)
    save(fig,"05_goal_set_paired_results","Paired outcomes show drawer recovery and original-goal regressions; oracle masking is a stronger drawer baseline.",[str(p.relative_to(ROOT)) for _,p in conditions])
    fig,axs=plt.subplots(1,2,figsize=(11.5,5))
    for ax,title,v,labs in [(axs[0],"Original drawer goal",[24,21],["Single goal","Learned goal set"]),(axs[1],"Reach-wall: arm pose matters",[23,0],["Released cost","Learned goal set"])]:
        bars=ax.bar([0,1],np.array(v)/24*100,color=[GRAY,RED],width=.6)
        ax.bar_label(bars,labels=[f"{x}/24" for x in v],padding=4)
        ax.set(xticks=[0,1],xticklabels=labs,ylim=(0,110),ylabel="Success rate (%)",xlabel="Goal-cost condition",title=title)
    foot(fig,"Exact simulator dynamics. Learned set K32, H16, magnitude 0.7. Drawer α=0; reach-wall α=0.1. These failures reject the current general-method claim.")
    fig.subplots_adjust(bottom=.2,wspace=.3)
    save(fig,"06_goal_set_failure_controls","No-harm and arm-relevance controls explain why ungated goal averaging is not a validated general method.",["docs/07_RESTART_LOG.md"])


def ranking(d):
    c=d["costs"];t=d["true_cost"];order=np.argsort(t);flips=np.zeros((5,5));i,j=np.triu_indices(len(t),1)
    for a in range(5):
        for b in range(5):flips[a,b]=np.mean(np.sign(c[a,i]-c[a,j])!=np.sign(c[b,i]-c[b,j]))
    fig,axs=plt.subplots(1,3,figsize=(14.4,4.8),gridspec_kw={"width_ratios":[1,1,1.15]})
    for ax,k,col in [(axs[0],0,BLUE),(axs[1],3,RED)]:
        ax.scatter(t,c[k],s=14,alpha=.6,color=col,edgecolors="none")
        ax.set(xlabel="True task cost (distance / success threshold)",ylabel="Visual latent cost (feature MSE)",title=f"Goal {k}: Spearman ρ = {spearmanr(t,c[k]).statistic:.2f}")
    im=axs[2].imshow(flips,cmap="magma",vmin=0,vmax=max(.5,flips.max()))
    for a in range(5):
        for b in range(5):axs[2].text(b,a,f"{100*flips[a,b]:.0f}%",ha="center",va="center",color="white" if flips[a,b]<.3 else INK,fontsize=10)
    axs[2].set(xticks=range(5),yticks=range(5),xlabel="Goal variant",ylabel="Goal variant",title="Candidate-pair ordering flips")
    fig.colorbar(im,ax=axs[2],fraction=.045,pad=.05,label="Fraction of pairwise rankings changed")
    foot(fig,"Drawer-close episode 002: same 200 candidate futures, exact dynamics, α=0. Illustrative episode, not an aggregate estimate.\nChanging only the goal reorders candidates; true task cost uses the same object target. Axes use measured values, not fitted predictions.")
    fig.subplots_adjust(bottom=.23,wspace=.42,top=.86)
    save(fig,"07_same_candidates_different_rankings","One saved episode illustrates changed candidate preferences under task-equivalent goals.",["source_data/drawer_offline_episode.npz"])


def font(size):
    for p in ["/System/Library/Fonts/Supplemental/Arial.ttf","/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists():return ImageFont.truetype(p,size)
    return ImageFont.load_default()


def animations(replay,drawer,push):
    size=288;gap=18;W=4*size+5*gap;H=445
    episode=np.load(DATA/"pusht_episode.npz");goal=episode["goal_state"]
    frames=[]
    modes=["wm","sim_latent","sim_true"]
    labels=["Learned model + latent cost","Exact dynamics + latent cost","Exact dynamics + true cost","Evaluation goal"]
    for step in range(len(replay["wm_frames"])):
        im=Image.new("RGB",(W,H),"white");dr=ImageDraw.Draw(im)
        dr.text((gap,10),"Push-T: replaying three saved plans from the same start",font=font(25),fill=INK)
        for j in range(4):
            x=gap+j*(size+gap)
            fr=replay[f"{modes[j]}_frames"][step] if j<3 else replay["goal_frame"]
            im.paste(Image.fromarray(fr).resize((size,size),Image.Resampling.NEAREST),(x,80))
            dr.text((x,52),labels[j],font=font(16),fill=[GRAY,BLUE,GREEN,INK][j])
            gx=x+goal[0]/512*size;gy=80+goal[1]/512*size;r=10
            dr.ellipse((gx-r,gy-r,gx+r,gy+r),outline=GREEN,width=2)
            if j<3:
                st=replay[f"{modes[j]}_states"][step];err=np.linalg.norm(st[:2]-goal[:2]);be=np.linalg.norm(st[2:4]-goal[2:4])
                dr.text((x,375),f"Pusher error {err:.1f}px · block {be:.1f}px",font=font(15),fill=INK)
                if step==len(replay["wm_frames"])-1:dr.text((x,397),"Final: "+("SUCCESS" if episode[f"{modes[j]}_success"] else "FAILURE"),font=font(17),fill=GREEN if episode[f"{modes[j]}_success"] else RED)
        dr.text((gap,425),f"Recorded episode 002 · env step {step}/30 · circle = target pusher position · target block overlay aligned to evaluation goal",font=font(14),fill=GRAY)
        frames.append(im)
    durations=[100]*len(frames);durations[0]=1000;durations[-1]=2600
    frames[0].save(OUT/"08_PushT_saved_plan_replay.gif",save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=False)
    frames[-1].save(OUT/"08_PushT_replay_final.png")
    manifest["figures"]["08_PushT_saved_plan_replay"]={"caption":"Genuine simulator replay of saved JEPA-WM, exact/latent and exact/true plans; selected illustrative failure episode 002, not a representative success-rate estimate.","sources":["source_data/pusht_episode.npz","source_data/replay_validation.json"],"note":"Target block overlay is aligned to evaluation goal; physics is unchanged."}
    for task,z in [("drawer",drawer),("push",push)]:
        ff=[]
        for k in [0,1,2,3,4]:
            im=Image.new("RGB",(760,455),"white");dr=ImageDraw.Draw(im)
            dr.text((18,12),f"Metaworld {task}: equivalent goal images",font=font(25),fill=INK)
            for j,v in enumerate([0,k]):
                x=18+j*375;im.paste(Image.fromarray(z["goal_frames"][v]).resize((350,350)),(x,70))
                dr.text((x,47),"Original expert goal" if j==0 else f"Goal variant {k}",font=font(18),fill=INK)
            dr.text((18,430),"Saved counterfactual renders · object outcome held fixed · not a time rollout",font=font(17),fill=GRAY)
            ff.append(im)
        ff[0].save(OUT/f"09_Metaworld_{task}_goal_variants.gif",save_all=True,append_images=ff[1:],duration=1200,loop=0)
        manifest["figures"][f"09_Metaworld_{task}_goal_variants"]={"caption":"Actual saved counterfactual goal frames, explicitly not a temporal rollout.","sources":[f"source_data/{task}_offline_episode.npz"]}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    replay=np.load(DATA/"pusht_replay.npz");drawer=np.load(DATA/"drawer_offline_episode.npz");push=np.load(DATA/"push_offline_episode.npz")
    pipeline(replay,drawer);benchmark_results();goal_set_diagram(drawer)
    for task,z in [("drawer",drawer),("push",push)]:
        goal_figures(task,z);heatmaps(task,np.load(DATA/f"{task}_dino_maps.npz"))
    method_results();ranking(drawer);animations(replay,drawer,push)
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("Created",len(manifest["figures"]),"visuals in",OUT)


def write_gallery():
    m=json.loads((OUT/"manifest.json").read_text())
    names={
        "01_project_overview":("Project overview","overview"),
        "02_oracle_attribution_results":("Where planning fails","results"),
        "03_drawer_equivalent_goals":("Drawer-close: equivalent goals","environments"),
        "03_push_equivalent_goals":("Metaworld push: equivalent goals","environments"),
        "04_drawer_DINO_patch_cost":("Drawer: measured DINOv2 patch costs","environments"),
        "04_push_DINO_patch_cost":("Push: measured DINOv2 patch costs","environments"),
        "05_goal_set_paired_results":("Paired outcomes for goal averaging","results"),
        "06_goal_set_failure_controls":("Why the tested goal-set idea fails","results"),
        "07_same_candidates_different_rankings":("Same futures, different goal rankings","results"),
        "08_PushT_saved_plan_replay":("Push-T: genuine saved-plan replay","environments"),
        "09_Metaworld_drawer_goal_variants":("Drawer goal-variation animation","environments"),
        "09_Metaworld_push_goal_variants":("Push goal-variation animation","environments"),
        "10_tested_goal_set_idea":("How the tested goal-set idea works","overview"),
    }
    order=["01_project_overview","10_tested_goal_set_idea","08_PushT_saved_plan_replay","03_drawer_equivalent_goals","04_drawer_DINO_patch_cost","03_push_equivalent_goals","04_push_DINO_patch_cost","09_Metaworld_drawer_goal_variants","09_Metaworld_push_goal_variants","02_oracle_attribution_results","05_goal_set_paired_results","06_goal_set_failure_controls","07_same_candidates_different_rankings"]
    sections=[]
    for key in order:
        title,cat=names[key];item=m["figures"][key]
        ext="gif" if (OUT/f"{key}.gif").exists() else "png"
        links=[f'<a href="{key}.{ext}" download>Download {ext.upper()}</a>']
        if (OUT/f"{key}.svg").exists():links.append(f'<a href="{key}.svg" download>Editable SVG</a>')
        if (OUT/f"{key}.mp4").exists():links.append(f'<a href="{key}.mp4" download>MP4</a>')
        sections.append(f'<section data-category="{cat}"><h2>{html.escape(title)}</h2><p>{html.escape(item["caption"])}</p><a href="{key}.{ext}"><img loading="lazy" src="{key}.{ext}" alt="{html.escape(title)}"></a><div class="links">'+" · ".join(links)+f'</div><details><summary>Sources and interpretation</summary><p>{html.escape(item.get("note",""))}</p><ul>'+"".join(f'<li><code>{html.escape(s)}</code></li>' for s in item["sources"])+"</ul></details></section>")
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>When the goal image says more than the task — visual evidence</title>
<style>body{font:16px/1.6 system-ui,sans-serif;color:#202733;background:#fff;margin:0}main{max-width:1280px;margin:auto;padding:32px 22px}h1{font-size:32px;line-height:1.2;max-width:900px}h2{font-size:23px;margin-bottom:4px}p{max-width:1000px}header p,.small{color:#596475}nav{display:flex;gap:8px;flex-wrap:wrap;padding:12px 0 24px}button{font:inherit;padding:8px 14px;border:1px solid #b4bdc9;background:white;border-radius:4px;cursor:pointer}button[aria-pressed=true]{background:#202733;color:white}section{padding:22px 0 32px;border-top:1px solid #dce1e7}section img{display:block;width:100%;height:auto;margin:18px 0}a{color:#235f94}.links{font-size:15px}details{margin-top:12px;font-size:14px;color:#596475}code{overflow-wrap:anywhere}.note{padding:14px 18px;border-left:3px solid #7b699e;background:#f6f4f9}section[hidden]{display:none}@media(prefers-reduced-motion:reduce){.small{color:#596475}}@media(max-width:600px){h1{font-size:26px}main{padding:20px 12px}}</style>
<main><header><p class="small">JEPA-WM project · 16 October 2026 · Evidence from existing saved runs</p><h1>When the goal image says more than the task</h1><p>A visual guide to the pipeline, real simulator scenes, measured feature costs, and the experiments completed so far.</p><p class="note"><strong>Scientific status:</strong> goal-image sensitivity is established on drawer-close. Ungated goal averaging improves some drawer cases but fails its general-method tests. This gallery does not present a validated new method.</p><p>Push-T animation replays saved plans and exactly matches their recorded final states. Metaworld animations cycle actual counterfactual goal renders; they are not time rollouts. DINO heatmaps show feature-MSE contributions, not attention or causal saliency.</p></header><nav aria-label="Filter figures">'''
    for value,label in [("all","All visuals"),("overview","Project map"),("environments","Environments & heatmaps"),("results","Results")]:
        page+=f'<button type="button" data-filter="{value}" aria-pressed="{str(value=="all").lower()}">{label}</button>'
    page+='</nav>'+"\n".join(sections)+'''<footer><p class="small">The images are real simulator renders and deterministic plots, with no AI-generated environment imagery. Open README.md and manifest.json for methods, caveats, and provenance. Vector figures use editable text and embedded scene images.</p></footer></main><script>document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-filter]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));document.querySelectorAll('[data-category]').forEach(s=>s.hidden=b.dataset.filter!=='all'&&s.dataset.category!==b.dataset.filter)}));</script></html>'''
    (OUT/"index.html").write_text(page)
    print("Gallery written",OUT/"index.html")


if __name__=="__main__":
    import sys
    if "--gallery-only" not in sys.argv:main()
    write_gallery()
