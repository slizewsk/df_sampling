"""Tests for `sampling` package."""

from df_sampling.core_imports import *
from df_sampling import *

class TestParams:   
    def setup_method(self):
        self.params = Params(phi0=1.0, gamma=0.5, alpha=3.5, beta=0.5, rmin=0.1, nsim=1000)

    def test_init(self):
        assert self.params.phi0 == 1.0
        assert self.params.gamma == 0.5
        assert self.params.alpha == 3.5
        assert self.params.beta == 0.5
        assert self.params.rmin == 0.1

    def test_calc_mvir(self):
        self.params.calculate_Mvir()
        assert self.params.Mvir ==  self.params.Mr(self.params.rvir)

    def test_draw_r(self):
        rvals = self.params.draw_r(n=10)
        assert len(rvals) == 10

    # def test_vbound(self):
    #     assert np.all(np.isfinite(self.params.vbound(self.params.draw_r(n=10))))

    def test_df(self):
        dfvals = self.params.df([0,1,1])
        assert np.isfinite(dfvals)

class TestParamsHernquist:
    def setup_method(self):
        self.params = ParamsHernquist(Mtot=1.0, a=10.0, beta = 0.0, rmin=0.1, nsim=1000)
    
    def test_init(self):
        assert self.params.Mtot == 1.0
        assert self.params.a == 10.0
        assert self.params.beta == 0.0
        assert self.params.rmin == 0.1
        assert self.params.nsim == 1000
    
    def test_draw_r(self):
        rvals = self.params.draw_r(n=10)
        assert len(rvals) == 10

    # def test_vbound(self):
    #     assert np.all(np.isfinite(self.params.vbound(self.params.draw_r(n=10))))

# class TestDataSampler:
#     @pytest.fixture
#     def mock_params(self):
#         """Creates a mock Params object with necessary methods."""
#         params = MagicMock(spec=Params)
#         params.nsim = 100
#         params.rmin = 0.1
#         params.rmax = 100
#         params.draw_r = MagicMock(return_value=np.linspace(0.1, 100, 100))
#         params.vbound = MagicMock(return_value=300)
#         params.phi = MagicMock(side_effect=lambda r: -1/r)
#         params.logdf = MagicMock(return_value=-5)
#         return params

#     @pytest.fixture
#     def sampler(self, mock_params):
#         """Creates a DataSampler instance with the mock Params object."""
#         return DataSampler(mock_params)

#     def test_initialization(self, sampler, mock_params):
#         """Test if DataSampler initializes correctly."""
#         assert sampler.params == mock_params
#         assert sampler.obsdf is None
#         assert sampler.calc_beta is None

#     def test_draw_positions(self, sampler):
#         """Test position drawing method."""
#         rvals = sampler.draw_positions(n=50)
#         assert len(rvals) == 50
#         assert np.all(rvals >= sampler.params.rmin)
#         assert np.all(rvals <= sampler.params.rmax)

#     def test_calculate_thresholds(self, sampler):
#         """Test threshold calculation."""
#         rvals = np.linspace(0.1, 10, 10)
#         thresholds = sampler.calculate_thresholds(rvals)
#         assert len(thresholds) == len(rvals)
#         assert np.all(np.isfinite(thresholds))

#     def test_sample_velocities(self, sampler):
#         """Test velocity sampling function."""
#         rvals = np.linspace(0.1, 10, 10)
#         threshs = np.full_like(rvals, -3)
#         velocities = sampler.sample_velocities(rvals, threshs)
#         assert velocities.shape == (10, 5)
#         assert np.all(np.isfinite(velocities))

#     def test_create_dataframe(self, sampler):
#         """Test DataFrame creation from sampled data."""
#         obs = np.random.rand(10, 5)
#         df = sampler.create_dataframe(obs)
#         assert isinstance(df, pd.DataFrame)
#         assert set(df.columns) == {'r', 'vr', 'vt', 'vtheta', 'vphi', 'v', 'r_err', 'vr_err', 'vt_err'}
#         assert len(df) == 10

#     def test_compute_beta(self, sampler):
#         """Test velocity anisotropy calculation."""
#         obs = np.random.rand(10, 5)
#         sampler.create_dataframe(obs)
#         beta = sampler.compute_beta()
#         assert np.isfinite(beta)


# class TestSampler:
#     def setup_method(self):
#         self.params = Params(phi0=50.0, gamma=0.5, alpha=3.3, beta=0.0, rmin=0.1, nsim=1000)
#         self.datasampler = DataSampler(self.params)

#     def test_draw_positions(self):
#         # Test w default n (should be self.params.ntracers)
#         rvals = self.datasampler.draw_positions()
#         assert isinstance(rvals, list), "Returned values should be a numpy array"
#         assert len(rvals) == self.params.nsim, f"Expected {self.params.nsim} samples, got {len(rvals)}"
#         assert np.all(np.array(rvals) >= self.params.rmin), "All drawn positions should be at least rmin"

#         # Test w custom n
#         n_custom = 500
#         rvals_custom = self.datasampler.draw_positions(n=n_custom)
#         assert len(rvals_custom) == n_custom, f"Expected {n_custom} samples, got {len(rvals_custom)}"
#         assert np.all(np.array(rvals_custom) >= self.params.rmin), "All drawn positions should be at least rmin"

#     def test_calcthresh(self):
#         rvals = self.datasampler.draw_positions()  # Generate rvals
#         threshs = self.datasampler.calculate_thresholds(rvals)  # Compute thresholds
            
#         assert isinstance(threshs, list), "Thresholds should be returned as a list"
#         assert len(threshs) == len(rvals), "Thresholds list should have the same length as rvals"

#     def test_velsamp(self):
#         rvals = self.datasampler.draw_positions()  # Generate rvals
#         threshs = self.datasampler.calculate_thresholds(rvals)  # Compute thresholds
#         obs = self.datasampler.sample_velocities(rvals,threshs)
#         assert isinstance(obs, np.ndarray), "Output should be a numpy array"
#         assert obs.shape == (len(rvals), 5), f"Expected shape ({len(rvals)}, 5), but got {obs.shape}"
#         assert np.all(np.isfinite(obs)), "All sampled velocity values should be finite" 


#     def test_create_dataframe(self):
#         rvals = self.datasampler.draw_positions()  
#         threshs = self.datasampler.calculate_thresholds(rvals)  
#         obs = self.datasampler.sample_velocities(rvals, threshs)  
#         self.datasampler.create_dataframe(obs)  # Create DataFrame

#         df = self.datasampler.obsdf  # Retrieve the DataFrame

#         assert isinstance(df, pd.DataFrame), "obsdf should be a pandas DataFrame"
#         expected_columns = {'r', 'vr', 'vt', 'vtheta', 'vphi', 'v'}
#         assert expected_columns.issubset(df.columns), f"DataFrame missing expected columns: {expected_columns - set(df.columns)}"
#         assert len(df) == len(obs), f"DataFrame should have {len(obs)} rows, but has {len(df)}"
#         assert df.notna().all().all(), "DataFrame should not contain NaN values"

#     def test_compute_beta(self):
#         rvals = self.datasampler.draw_positions()  
#         threshs = self.datasampler.calculate_thresholds(rvals)  
#         obs = self.datasampler.sample_velocities(rvals, threshs)  
#         self.datasampler.create_dataframe(obs)  
#         self.datasampler.compute_beta()  

#         assert hasattr(self.datasampler, 'calc_beta'), "calc_beta attribute should be set after computing beta"
#         assert np.isfinite(self.datasampler.calc_beta), "Computed beta should be a finite number"
   
#     def test_run_sampling(self):
#         self.datasampler.run_sampling()  

#         assert isinstance(self.datasampler.obsdf, pd.DataFrame), "obsdf should be a pandas DataFrame after run_sampling"
#         assert 'r' in self.datasampler.obsdf.columns, "obsdf should contain column 'r'"
#         assert 'vr' in self.datasampler.obsdf.columns, "obsdf should contain column 'vr'"
#         assert 'vt' in self.datasampler.obsdf.columns, "obsdf should contain column 'vt'"
#         assert 'vtheta' in self.datasampler.obsdf.columns, "obsdf should contain column 'vtheta'"
#         assert 'vphi' in self.datasampler.obsdf.columns, "obsdf should contain column 'vphi'"
#         assert len(self.datasampler.obsdf) > 0, "obsdf should not be empty after run_sampling"
        
#         assert hasattr(self.datasampler, 'calc_beta'), "calc_beta should be set after run_sampling"
#         assert np.isfinite(self.datasampler.calc_beta), "Computed beta should be a finite number"


# # # not sure if i need to / shoudl also test indicv fcns, or if using the final sampler class that incorps everything is ok
# # # for example, here i tested draw_positions that calls draw_r function. do i need to test the draw_rs fcn directly now?
