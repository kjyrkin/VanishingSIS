import numpy as np
import itertools
import csv
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler as SampleD

N = 32
q = 1048897 # Choose q such that it is a prime and equal to 1 mod 2*N

K = CyclotomicField(N*2)
R = K.ring_of_integers()

# R/qR is isomorphic to the ring S = ZZ_q[X] / (x^N + 1) via the coefficient embedding;
# we use S to check if elements of R/qR are units
Rtmp.<x> = GF(q)[]
S = Rtmp.quotient(x^N + 1,'x')

# Sample an element in R; the distribution of coefficients is determined by the parameter
# 'sampler'. If ensure_unit=True we check that the output element is a unit modulo q.
def sample_R(sampler, ensure_unit=False):
    if not ensure_unit:
        return R([sampler() for _ in range(S.degree())])
    while True:
        arr = [sampler() for _ in range(S.degree())]
        if S(arr).is_unit():
            return R(arr)
    

# 2-norm of module element in coefficient embedding
def coeff_norm_mod(a):
    return vector(itertools.chain.from_iterable([elem.list() for elem in a])).norm()

# Compute the "centered moving median" of (x,y) data, with some window size.
# Assumes data is sorted according to x-coordinates
def moving_median(arr, window_size):
    result = []
    xwindow = []
    ywindow = []
    for x,y in arr:
        xwindow.append(x)
        ywindow.append(y)
        if len(xwindow) >= window_size:
            result.append((np.median(xwindow),np.median(ywindow)))
            xwindow.pop(0)
            ywindow.pop(0)
    return result

# Compute the leftmost intercept of a curve (defined by (x,y) tuples, sorted by the x-coordinate)
# and the line y=x.
def compute_intercept(curve):
    for i in range(len(curve)):
        x1,y1 = curve[i]
        if x1 > y1:
            x0,y0 = curve[i-1]
            # 'nearest neighbour'-interpolation
            if abs(x0-y0) <= abs(x1-y1):
                return x0
            return x1

# Run the numerical experiments for some vSIS degree D, with nsamples many samples 
def compute_samples(D, nsamples, print_progress=False):
    data = []
    ldev = 0.8 # parameter in (0,1)
    rdev = 0.97

    rootq = q^(1/(D+1))
    
    counter = 0
    progress = 0

    for param in np.linspace(((1-ldev)*rootq).n(),((1+rdev)*rootq).n(),num=nsamples):
        sigma_f = 1.17 * param/sqrt(2*N)
        sampler = SampleD(sigma_f)
        
        f = sample_R(sampler)
        g = sample_R(sampler, True)

        b1norm = coeff_norm_mod([f,g]).n()

        fstar = f.conjugate()
        gstar = g.conjugate()
        c = q / sum([(f*fstar)^(D-k)*(g*gstar)^k for k in range(D+1)])
        bN = [c*x for x in [fstar^k*gstar^(D-k) for k in range(D+1)]]

        bNnorm = coeff_norm_mod(bN).n()

        data.append((b1norm,bNnorm))
        
        if print_progress:
            counter += 1
            ptmp = floor((counter / nsamples)*10)/10
            if ptmp > progress:
                progress = ptmp
                print('{}% ...'.format(progress*100))
    
    return sorted(data)

# Helper symbolic variable
var('xp')

# Plot the data
def create_figure(data,D,xlim_coeff,ylim_coeff,windowsize):
    rootq = q^(1/(D+1))
    rootqlabel = r'\sqrt{q}' if D==1 else r'q^{1/'+str(D+1)+'}'
    
    xt = [i*rootq for i in range(20)]
    xt_lbl = ['$0$', '$'+rootqlabel+'$'] + ['$'+str(i)+rootqlabel+'$' for i in range(2,20)]

    yt = copy(xt)
    yt_lbl = copy(xt_lbl)
    
    if ylim_coeff >= 10:
        yt = yt[::2]
        yt_lbl = yt_lbl[::2]

    mm = moving_median(data,windowsize)
    icpt = compute_intercept(mm)
    gs_slack = RDF(icpt/rootq).n(digits=3)

    xt.append(icpt)
    lbl_offset = r'\;\qquad' if gs_slack < 1.2 else ''
    xt_lbl.append('$' + lbl_offset + str(gs_slack) + rootqlabel + '$')

    xlim = xlim_coeff*rootq
    ylim = ylim_coeff*rootq

    g = Graphics()

    g += list_plot(data, ticks=[xt, yt], tick_formatter=[xt_lbl, yt_lbl], xmin=0, xmax=xlim,
                   ymin=0, ymax=ylim, frame=True, axes=False, color='#003f5c',
                   legend_label=r'$\|\tilde{\mathbf{t}}_{'+str(D+1)+'}\|$',zorder=50)
    g += list_plot(mm, plotjoined=True, color='#ffa600', thickness=2.5,
                   zorder=51, legend_label=r'$\mathrm{median}$')
    den_exp = '^{}'.format(D) if D > 1 else ''
    g += plot(q/xp^D, (xp, 0, xlim+1000), color='black', legend_label=r'$q / \|\mathbf{t}_1\|'+den_exp+'$')
    g += plot(xp, (xp, 0, xlim+1000), color='black', legend_label=r'$\|\mathbf{t}_1\|$')
    g += line([(icpt,0),(icpt,ylim)], color='black', linestyle=':',title=r'$D = '+str(D)+'$'.format(D))
    
    g.fontsize(10)
    g.axes_labels([r'$\|\mathbf{t}_1\|$',''])
    g.axes_labels_size(1.2)
    
    legend_loc = 'upper right' if D==1 else 'lower left'
    g.set_legend_options(font_size=12, loc=legend_loc)

    return g

def csv_write(data,fname):
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
        
        
def csv_read(fname):
    with open(fname, newline='') as f:
        reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
        return [tuple(row) for row in reader]
    
    
nsamples = 2000
wsize = 250 # window size for the median
Dlow = 1
Dhi = 6

datafolder = 'data/'
datas = {}

# The actual computation (warning: rather slow)
for D in range(Dlow,Dhi+1):
    print('Computing for D={}'.format(D))
    data = compute_samples(D,nsamples,True)
    print('Done.')
    
    datas[D] = data
    fname = '{}data_D{}.csv'.format(datafolder,D)
    csv_write(data,fname)
    
# Plot the norm of last column for different norms of the first one:

lim_params = {1 : (2,6), 2 : (2,12), 3 : (2,14), 4 : (2,16)}
garr = []
figfolder = 'figs/'

for D in range(1,5):
    fname = '{}data_D{}.csv'.format(datafolder,D)
    data = csv_read(fname)
    
    xlimc,ylimc = lim_params[D]
    p = create_figure(data,D,xlimc,ylimc,wsize)
    p.fontsize(10)
    p.set_legend_options(font_size=10)
    garr.append(p)
    
g1 = graphics_array([[garr[0],garr[1]],
                     [garr[2],garr[3]]])
g1.show(figsize=[9, 7])
# Uncomment to save as eps:
g1.save('{}norm_last_col.eps'.format(figfolder), figsize=[9, 7])

# Plot the optimal GS-norms for different values of D

norms = []
gs_slacks = []

for D in datas.keys():
    mm = moving_median(datas[D],wsize)
    icpt = compute_intercept(mm)
    norms.append(icpt)
    rootq = q^(1/(D+1))
    gs_slacks.append(icpt/rootq)

p1 = list_plot(list(zip(datas.keys(),norms)),plotjoined=True,color='#003f5c',thickness = 2,
               marker='o',linestyle=':', frame=True, axes=False)
p1.axes_labels([r'$D$',r'$\mathrm{optimal \;\;} \|\mathbf{T}\|_{\mathrm{GS}}$'])

p2 = list_plot(list(zip(datas.keys(),norms)),plotjoined=True,color='#003f5c',thickness = 2,
               marker='o',linestyle=':', frame=True, axes=False, scale='semilogy')
p2.axes_labels([r'$D$',''])

g2 = graphics_array([p1,p2])
g2.show(figsize=[10,5])
g2.save('{}Tnorms_both.eps'.format(figfolder), figsize=[10,5])

# Plot gs_slack to visualize the trend
g3 = list_plot(list(zip(datas.keys(),gs_slacks)),plotjoined=True,color='#003f5c',thickness = 2,
               marker='o',linestyle=':', frame=True, axes=False)
g3.axes_labels([r'$D$',r'$\mathrm{GS\_SLACK}$'])
g3.show()
g3.save('{}gs_slack.eps'.format(figfolder))

