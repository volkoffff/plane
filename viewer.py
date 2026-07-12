import pyvista as pv
import aircraft as aircraft
import simulation as simulation
import time
import numpy as np


def import_aircraft_actors(plotter: pv.Plotter, model_path: str):
    model_center = np.array(pv.read(model_path).center, dtype=float)
    existing_actor_names = set(plotter.renderer.actors)
    plotter.import_gltf(model_path, set_camera=False)
    actors = [
        actor
        for name, actor in plotter.renderer.actors.items()
        if name not in existing_actor_names
    ]

    if not actors:
        raise RuntimeError(f"Aucun acteur charge depuis le modele 3D : {model_path}")

    return actors, model_center


def run_viewer() -> None:
    plotter = pv.Plotter()

    dt = 0.01

    state = simulation.initial_state()
    aircraft_params = simulation.rafale_like_parameters()
    speed_ctrl_state = simulation.SpeedControllerState()

    ##########
    # axis
    ##########
    plotter.show_axes() # show axis
    plotter.show_grid() # show grid

    ##########
    # ground
    ##########
    ground = pv.Plane(
        center=(0, 0, 0),
        direction=(0, 0, 1),
        i_size=200,
        j_size=200,
    )

    plotter.add_mesh(ground, color="lightgray", opacity=0.25)

    ##########
    # aircraft
    ##########
    aircraft_mesh = aircraft.create_aircraft() # model 3d

    # actor = plotter.add_mesh(aircraft_mesh) # add the model to the scene
    aircraft_actors, aircraft_center = import_aircraft_actors(plotter, str(aircraft_mesh)) # add the model to the scene

    ##########
    # speed chart
    ##########
    speed_chart = pv.Chart2D(size=(0.30, 0.20), loc=(0.03, 0.70))
    speed_chart.title = "Speed"
    speed_chart.x_label = "time (s)"
    speed_chart.y_label = "m/s"
    speed_plot = speed_chart.line([0.0], [np.linalg.norm(state.velocity)], color="blue")
    speed_plot.line_width = 2
    plotter.add_chart(speed_chart)

    plotter.show(interactive_update=True)

    trajectory = []
    speed_times = [0.0]
    speed_values = [float(np.linalg.norm(state.velocity))]
    elapsed_time = 0.0
    chart_time_window = 20.0

    alpha_target = np.deg2rad(5.0)

    altitude_target = 1000.0
    throttle_command = 0

    while True:
      alpha_target = simulation.altitude_controller(state, altitude_target)
      elevator_control = simulation.alpha_controller(state, alpha_target)
      state = simulation.update_state(state, aircraft_params, elevator_control, throttle_command, dt)
      trajectory.append(state.position.copy())
      elapsed_time += dt

      position = state.position - aircraft_center
      for actor in aircraft_actors:
          actor.SetPosition(float(position[0]), float(position[1]), float(position[2]))
          actor.RotateY(-state.pitch_rate)


      points = np.array(trajectory)

      if len(points) > 1:
        line = pv.lines_from_points(points)
        plotter.add_mesh(line)

      speed_times.append(elapsed_time)
      speed_values.append(float(np.linalg.norm(state.velocity)))

      while speed_times and elapsed_time - speed_times[0] > chart_time_window:
        speed_times.pop(0)
        speed_values.pop(0)

      speed_plot.update(np.array(speed_times), np.array(speed_values))

      plotter.update()
      time.sleep(dt)

    ##########
    # camera
    ##########
    plotter.camera_position = "iso"

    plotter.show()


if __name__ == "__main__":
    run_viewer()
