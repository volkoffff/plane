from dataclasses import dataclass

from physics.projectiles import Bullet, create_bullet_from_aircraft, integrate_bullets
from piloting.commands import AircraftAction


@dataclass
class Gun:
    fire_rate: float = 12.0
    cooldown: float = 0.0
    damage: float = 10.0
    ammo: int | None = None

    def update(self, dt: float) -> None:
        self.cooldown = max(0.0, self.cooldown - dt)

    def can_fire(self) -> bool:
        if self.cooldown > 0.0:
            return False
        return not (self.ammo is not None and self.ammo <= 0)

    def mark_fired(self) -> None:
        self.cooldown = 1.0 / self.fire_rate
        if self.ammo is not None:
            self.ammo -= 1

    def try_fire(self) -> bool:
        if not self.can_fire():
            return False

        self.mark_fired()
        return True

@dataclass
class WeaponSystem:
    def update(
        self,
        world,
        actions: dict[int, AircraftAction],
        dt: float,
    ) -> None:
        for entity in world.aircraft.values():
            entity.weapon.update(dt)

            if entity.is_dead():
                continue

            action = actions.get(entity.id)
            if action is None:
                continue

            if action.fire_gun and entity.weapon.try_fire():
                bullet = create_bullet_from_aircraft(
                    entity.physics.state,
                    owner_id=entity.id,
                    team=entity.team,
                    damage=entity.weapon.damage,
                )
                world.bullets.append(bullet)

        world.bullets = integrate_bullets(world.bullets, dt)
