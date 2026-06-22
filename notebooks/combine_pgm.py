import matplotlib.pyplot as plt, matplotlib.image as mpimg, matplotlib.gridspec as gridspec
OUT="hierarchical_poc_outputs"
# Figure A: JTF + FTJ side by side
imgs=[mpimg.imread(f"{OUT}/pgm_{n}.png") for n in ["jtf","ftj"]]
ratios=[im.shape[1]/im.shape[0] for im in imgs]
fig=plt.figure(figsize=(11,4.3)); gs=gridspec.GridSpec(1,2,width_ratios=ratios,wspace=0.05)
for i,im in enumerate(imgs):
    ax=fig.add_subplot(gs[i]); ax.imshow(im); ax.axis("off")
fig.savefig(f"{OUT}/pgm_jtf_ftj.png",dpi=170,bbox_inches="tight"); print("wrote pgm_jtf_ftj.png")
# Figure B: hierarchical alone, larger
im=mpimg.imread(f"{OUT}/pgm_hier.png")
fig=plt.figure(figsize=(8,6)); ax=fig.add_subplot(111); ax.imshow(im); ax.axis("off")
fig.savefig(f"{OUT}/pgm_hier_solo.png",dpi=190,bbox_inches="tight"); print("wrote pgm_hier_solo.png")
