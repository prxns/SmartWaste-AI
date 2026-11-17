"""Draw exact academic diagrams and calculated fuzzy figures with Matplotlib."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from config import ROOT, MEMBERSHIPS
from smartwaste.fuzzy_engine import infer, membership


def main():
    target=ROOT/"assets/figures"; target.mkdir(parents=True,exist_ok=True)
    result=infer(80,20)
    fig,ax=plt.subplots(figsize=(8,3.5))
    for name,color in zip(MEMBERSHIPS,("#839989","#AE9A36","#22694D")):
        ax.plot(result.universe,membership(result.universe,MEMBERSHIPS[name]),label=name,color=color,lw=2)
    ax.set(xlim=(0,100),ylim=(0,1.08),xlabel="Normalized input or output value",ylabel="Membership degree",title="Configured Low Medium and High membership functions")
    ax.legend(); ax.grid(alpha=.15); fig.tight_layout()
    fig.savefig(target/"memberships.png",dpi=180); plt.close(fig)
    result=infer(65,35)
    fig,ax=plt.subplots(figsize=(8,3.5))
    for name,values in result.outputs.items(): ax.plot(result.universe,values,ls=":",alpha=.5,label=name)
    ax.fill_between(result.universe,result.aggregate,color="#6D9964",alpha=.4,label="Aggregated output")
    ax.plot(result.universe,result.aggregate,color="#22694D",lw=2)
    ax.axvline(result.score,color="#203F2F",ls="--",label=f"Centroid {result.score:.2f}")
    ax.set(xlim=(0,100),ylim=(0,1.08),xlabel="Recycling suitability",ylabel="Membership degree",title="Calculated example with potential 65 and contamination 35")
    ax.legend(ncol=3,fontsize=8); fig.tight_layout()
    fig.savefig(target/"fuzzy_example.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6)); ax.set(xlim=(0,8),ylim=(0,6)); ax.axis("off")
    nodes=[(1,5,"JPEG or PNG image"),(1,3.8,"Validation and preprocessing"),
           (1,2.6,"MobileNetV2 material classification"),(1,1.4,"Configured category potential"),
           (5,2.6,"Manual contamination"),(5,1.4,"Mamdani min max inference"),
           (5,.2,"Centroid and guidance")]
    for x,y,label in nodes:
        ax.add_patch(FancyBboxPatch((x-.7,y),3,.65,boxstyle="round,pad=0.12",facecolor="#EAF0E7",edgecolor="#688C70"))
        ax.text(x+.8,y+.325,label,ha="center",va="center",fontsize=8.5)
    for start,end in [((1.8,5),(1.8,4.45)),((1.8,3.8),(1.8,3.25)),((1.8,2.6),(1.8,2.05)),((3.3,1.725),(4.3,1.725)),((5.8,2.6),(5.8,2.05)),((5.8,1.4),(5.8,.85))]:
        ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="->",color="#2A5940",lw=1.5))
    fig.tight_layout(); fig.savefig(target/"architecture.png",dpi=180); plt.close(fig)


if __name__=="__main__": main()
