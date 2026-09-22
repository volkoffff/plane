from dataclasses import dataclass, field


@dataclass(frozen=True)
class DamageEvent:
    target_id: int
    attacker_id: int | None
    damage: float
    reason: str 

@dataclass
class CombatSystem:
    events: list[DamageEvent] = field(default_factory=list)

    def update(self, world, dt: float) -> None:
        pass

    def apply_damage(
        self,
        target,
        damage: float,
        attacker_id: int | None = None,
        reason: str = "unknown",
    ) -> None:
        if not target.alive:
            return

        target.health = max(0.0, target.health - damage)

        self.events.append(
            DamageEvent(
                target_id=target.id,
                attacker_id=attacker_id,
                damage=damage,
                reason=reason,
            )
        )

        if target.health <= 0.0:
            target.death()
