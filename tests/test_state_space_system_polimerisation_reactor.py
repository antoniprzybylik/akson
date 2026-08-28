import pytest
import torch
import torch.nn as nn
import math
from akson import StateSpaceDynamics, StateSpaceSystem, OperatingPoint


class FModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.C_I_in = 8.0
        self.C_m_in = 6.0
        self.E_T_c = 2.9442e3
        self.E_T_d = 2.9442e3
        self.E_f_m = 7.4478e4
        self.E_I = 1.2550e5
        self.E_P = 1.8283e4
        self.Z_T_c = 3.8223e10
        self.Z_T_d = 3.1457e11
        self.Z_f_m = 1.0067e15
        self.Z_I = 3.7920e18
        self.Z_P = 1.7700e9
        self.f_star = 0.58
        self.V = 0.1
        self.M_m = 100.12
        self.R = 8.314
        self.T = 335.0
        self.F = 1.0
        self.eps = 1e-16

    def forward(self, t, x, u):
        C_m, C_I, D_0, D_I = x[0:1], x[1:2], x[2:3], x[3:4]
        F_I = u

        term_init = self.Z_I * math.exp(-self.E_I / (self.R * self.T))
        term_prop_tfm = (self.Z_P * math.exp(-self.E_P / (self.R * self.T))) + (
            self.Z_f_m * math.exp(-self.E_f_m / (self.R * self.T))
        )

        p0_denominator = (self.Z_T_d * math.exp(-self.E_T_d / (self.R * self.T))) + (
            self.Z_T_c * math.exp(-self.E_T_c / (self.R * self.T))
        )
        p0_numerator = 2 * self.f_star * torch.clamp(C_I, min=self.eps) * term_init
        P_0 = torch.sqrt(p0_numerator / (p0_denominator + self.eps))

        dC_I_dt = (
            -term_init * C_I - (self.F / self.V) * C_I + (F_I * self.C_I_in / self.V)
        )
        dC_m_dt = (
            -term_prop_tfm * C_m * P_0
            - (self.F / self.V) * C_m
            + (self.F * self.C_m_in / self.V)
        )

        term_term_d0 = (
            0.5 * self.Z_T_c * math.exp(-self.E_T_c / (self.R * self.T))
        ) + (self.Z_T_d * math.exp(-self.E_T_d / (self.R * self.T)))
        term_tfm_d0 = self.Z_f_m * math.exp(-self.E_f_m / (self.R * self.T))
        dD_0_dt = (
            term_term_d0 * P_0**2 + term_tfm_d0 * C_m * P_0 - (self.F / self.V) * D_0
        )

        dD_I_dt = self.M_m * term_prop_tfm * C_m * P_0 - (self.F / self.V) * D_I

        return torch.cat([dC_m_dt, dC_I_dt, dD_0_dt, dD_I_dt], dim=0)


class GModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-16

    def forward(self, x, u):
        D_0 = x[2:3]
        D_I = x[3:4]
        namw = D_I / (D_0 + self.eps)
        return namw


@pytest.fixture
def reactor_dynamics():
    FI_min = 0.003
    FI_max = 0.06
    return StateSpaceDynamics(
        FModule(),
        GModule(),
        n_inputs=1,
        n_outputs=1,
        state_size=4,
        u_min=torch.tensor([FI_min], dtype=torch.float64),
        u_max=torch.tensor([FI_max], dtype=torch.float64),
        dtype=torch.float64,
        device=torch.device("cpu"),
        max_spectral_radius=11.7,
    )


def test_operating_point_from_input_reactor(reactor_dynamics):
    FI0 = 1.678300e-02
    op = OperatingPoint.from_input(
        reactor_dynamics, torch.tensor([FI0], dtype=torch.float64)
    )

    # Verify that equilibrium point was found
    t0 = torch.as_tensor(0.0, dtype=torch.float64)
    with torch.no_grad():
        residual = reactor_dynamics.f_original(t0, op.x, op.u)
    assert residual.norm().item() < 1e-5

    # Verify that the output has physical sense (approx. 25000)
    assert 20000 < op.y.item() < 30000


def test_reactor_simulation_bounds(reactor_dynamics):
    FI0 = 1.678300e-02
    op = OperatingPoint.from_input(
        reactor_dynamics, torch.tensor([FI0], dtype=torch.float64)
    )
    reactor_dynamics.operating_point = op
    system = StateSpaceSystem(reactor_dynamics)

    def bad_u(t):
        return torch.tensor([0.1], dtype=torch.float64)  # Exceeds FI_max

    with pytest.raises(ValueError, match="Input \\[0.1\\] exceeds u_max \\[0.06\\]."):
        system.simulate(bad_u, duration=1.0, step_size=0.1)


def test_reactor_continuous_simulation(reactor_dynamics):
    FI0 = 1.678300e-02
    op = OperatingPoint.from_input(
        reactor_dynamics, torch.tensor([FI0], dtype=torch.float64)
    )
    reactor_dynamics.operating_point = op
    system = StateSpaceSystem(reactor_dynamics)

    def u1(t):
        return torch.tensor([FI0], dtype=torch.float64)

    t1, x1, y1 = system.simulate(u1, duration=1.0, step_size=0.1)

    state_after_first = system.x.clone()

    def u2(t):
        return torch.tensor([0.003], dtype=torch.float64)  # Now, feed in FI_min

    t2, x2, y2 = system.simulate(u2, duration=1.0, step_size=0.1)

    # Continuity of timepoints and the state
    assert t2[0].item() == pytest.approx(1.0)
    assert torch.allclose(x2[0], state_after_first, atol=1e-8)
