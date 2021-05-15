# We are dealing with the "Young's Double Slit Experiment" simulation here:
# https://github.com/NSLS-II/sirepo-bluesky/blob/master/sirepo_bluesky/tests/SIREPO_SRDB_ROOT/user/SdP3aU5G/srw/00000000/sirepo-data.json

RE(
    optimization_plan(
        fly_plan=run_fly_sim,
        bounds=param_bounds,
        db=db,
        run_parallel=True,
        num_interm_vals=1,
        num_scans_at_once=2,
        sim_id="00000000",
        server_name="http://localhost:8000",
        root_dir=root_dir,
        watch_name="W60",
        flyer_name="sirepo_flyer",
        intensity_name="mean",
        opt_type="sirepo",
        max_iter=5,
    )
)
