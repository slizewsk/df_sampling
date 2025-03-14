Usage
=====

Here are some example usages of `df_sampling`.

Usage Example: Sampling with `ParamsHernquist` and `DataSampler`
=================================================================

This example demonstrates how to use the `ParamsHernquist` class 
and `DataSampler` to run a sampling process, plot the results, 
and calculate the specific energy for the sampled data.

Step 1: Scaling and Setting up Parameters
-------------------------------------------
Before running the sampling, we scale the mass from `G=1` to `1e12 Msun`. 
We do this by multiplying by a constant factor `2.325e-3`. 
The scaled mass is used to initialize the `ParamsHernquist` object.

```
python
params_h = ParamsHernquist(Mtot=scale_to_g1_mass(1), 
                           a=10,
                           beta=0.0,
                           nsim=140,
                           rmin=1e-4,
                           test_dir='Mar13Hernquist')
```

Step 2: Running the Sampling
------------------------------
We initialize a `DataSampler` object with the `ParamsHernquist` object 
and then run the sampling process to generate velocities for the given configuration.
```
DS = DataSampler(params_h)
vels = DS.run_sampling(verbose=False)
```
Step 3: Plotting the Results
------------------------------
We visualize the sampled points using a log-scaled plot for galactocentric distance and speed.

```
fig, axs = plt.subplots(figsize=(6, 3))

rs = np.logspace(np.log10(min(DS.obsdf.r)), 3.5, 300)
axs.plot(rs, DS.params.vbound(rs), label=None, color='orange', ls='--')

axs.scatter(DS.obsdf.r, DS.obsdf.v, marker='o', s=5, alpha=0.4, label='Sampled Points', rasterized=True)

axs.set_xscale('log')
axs.set_xlabel('Galactocentric Distance [kpc]')
axs.set_ylabel('Total Speed [100 km/s]')
plt.show()
```
