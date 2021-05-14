RE(
    optimization_plan(
        fly_plan=run_fly_sim,
        bounds=param_bounds,
        db=db,
        run_parallel=True,
        num_interm_vals=1,
        num_scans_at_once=2,
        sim_id="Gu73sx1B",
        server_name="http://localhost:8000",
        root_dir=root_dir,
        watch_name="W60",
        flyer_name="sirepo_flyer",
        intensity_name="mean",
        opt_type="sirepo",
    )
)
