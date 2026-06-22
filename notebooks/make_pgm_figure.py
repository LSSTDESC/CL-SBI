"""
Render PGMs for JTF / FTJ / Hierarchical via numpyro.render_model, with graphviz HTML-like labels
giving true Greek glyphs + proper subscripts. Structural-only models (NFW forward = trivial op) so
graph topology matches the real models without heavy computation.
"""
import warnings; warnings.filterwarnings("ignore")
import re, numpy as np, jax.numpy as jnp, numpyro
import numpyro.distributions as dist
OUT = "/Users/akumgill/Documents/GitHub/CL-SBI/notebooks/hierarchical_poc_outputs"
N_c, NR = 376, 30
sig = jnp.ones(NR)*0.3
y_stack = jnp.zeros(NR); y_obs = jnp.zeros((N_c, NR))
def fwd1(logM,c): return jnp.broadcast_to((logM+c)*0.0,(NR,))
def fwdN(logM,c): return (logM[:,None]+c[:,None])*0.0 + jnp.zeros((1,NR))

def model_jtf(y=None):
    logM=numpyro.sample("log10M",dist.Uniform(13.,16.)); c=numpyro.sample("c",dist.Uniform(2.,9.))
    lnf=numpyro.sample("lnf",dist.Uniform(-10.,10.))
    s=jnp.sqrt(sig**2+jnp.exp(lnf)**2)
    numpyro.sample("Sigma_stacked",dist.Normal(fwd1(logM,c),s),obs=y)

def model_ftj(y=None):
    logM=numpyro.sample("log10M",dist.Uniform(13.,16.)); c=numpyro.sample("c",dist.Uniform(2.,9.))
    lnf=numpyro.sample("lnf",dist.Uniform(-10.,10.))
    s=jnp.sqrt(sig**2+jnp.exp(lnf)**2); mu=fwd1(logM,c)
    with numpyro.plate("clusters",N_c):
        numpyro.sample("Sigma_j",dist.Normal(jnp.broadcast_to(mu,(N_c,NR)),s).to_event(1),obs=y)

def model_hier(y=None):
    muM=numpyro.sample("muM",dist.Normal(14.4,.4)); sigM=numpyro.sample("sigM",dist.HalfNormal(.4))
    c0=numpyro.sample("c0",dist.Normal(4.6,1.5)); beta=numpyro.sample("beta",dist.Normal(0.,2.))
    sigc=numpyro.sample("sigc",dist.HalfNormal(.7)); lnf=numpyro.sample("lnf",dist.Uniform(-10.,10.))
    s=jnp.sqrt(sig**2+jnp.exp(lnf)**2)
    with numpyro.plate("clusters",N_c):
        logM=numpyro.sample("log10M_j",dist.Normal(muM,sigM))
        c=numpyro.sample("c_j",dist.Normal(c0+beta*(logM-muM),sigc))
        numpyro.sample("Sigma_j",dist.Normal(fwdN(logM,c),s).to_event(1),obs=y)

# HTML-like labels: &#956;=mu  &#963;=sigma  &#946;=beta  &#931;=Sigma
LBL = {
    "log10M":        "log<SUB>10</SUB>M",
    "c":             "c",
    "lnf":           "ln f",
    "Sigma_stacked": "&#931;<SUB>stack</SUB>",
    "Sigma_j":       "&#931;<SUB>j</SUB>",
    "muM":           "&#956;<SUB>M</SUB>",
    "sigM":          "&#963;<SUB>M</SUB>",
    "c0":            "c<SUB>0</SUB>",
    "beta":          "&#946;",
    "sigc":          "&#963;<SUB>c</SUB>",
    "log10M_j":      "log<SUB>10</SUB>M<SUB>j</SUB>",
    "c_j":           "c<SUB>j</SUB>",
}
def relabel(line):
    def repl(m):
        node=m.group(1); disp=LBL.get(node, node)
        return f'{node} [label=<{disp}>'
    return re.sub(r'(\w+) \[label=\w+', repl, line)

for name,model,y,title in [("jtf",model_jtf,y_stack,"Join-then-fit (JTF)"),
                           ("ftj",model_ftj,y_obs,"Fit-then-join (FTJ)"),
                           ("hier",model_hier,y_obs,"Hierarchical (this work)")]:
    g=numpyro.render_model(model,model_args=(y,),render_params=True)
    g.body=[relabel(l) for l in g.body]
    g.attr(label=title,labelloc="t",fontsize="20")
    g.render(f"{OUT}/pgm_{name}",format="pdf",cleanup=True)
    g.render(f"{OUT}/pgm_{name}",format="png",cleanup=True)
    print("wrote pgm_"+name)
print("done")
