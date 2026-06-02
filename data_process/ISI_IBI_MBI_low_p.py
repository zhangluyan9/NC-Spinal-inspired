import numpy as np
import pandas as pd

def simulate_neuro_circuit(branches):
    Roff = 64180
    Ron = 1500
    Vth = 1.4
    Vhold = 0.9
    R0_fusion = 40
    R0_output = 100
    gain = 45
    T_total = 2000e-6
    N = 50_000
    t = np.linspace(0, T_total, N)

    def parallel(R1, R2):
        return (R1 * R2) / (R1 + R2)

    def Vm_and_Rt(t, RL, Roff, Ron, Vin, Vth, Vhold, Cm):
        Vm = np.zeros_like(t)
        Rt = np.zeros_like(t)
        if abs(Vin) < 1e-6:
            return Vm, Rt + 1e12
        Vth_l = np.sign(Vin) * Vth
        Vhold_l = np.sign(Vin) * Vhold
        V_ss = Ron / (RL + Ron) * Vin
        tau_f = parallel(RL, Ron) * Cm
        tau_r = parallel(RL, Roff) * Cm
        tr = tau_r * np.log((Roff/(RL+Roff)*(Vin - Vhold_l)) / (Roff/(RL+Roff)*Vin - Vth_l))
        tf = tau_f * np.log((Vth_l - V_ss)/(Vhold_l - V_ss))
        period = tr + tf

        for i, ti in enumerate(t):
            tp = ti % period
            if tp <= tr:
                Vm[i] = (Roff/(RL+Roff)*Vin) - ((Roff/(RL+Roff)*Vin) - Vhold_l)*np.exp(-tp/tau_r)
                Rt[i] = Roff
            else:
                tp2 = tp - tr
                Vm[i] = (Ron/(RL+Ron)*Vin) - ((Ron/(RL+Ron)*Vin) - Vth_l)*np.exp(-tp2/tau_f)
                Rt[i] = Ron

        return Vm, Rt

    def Vm_and_Rt_for_signal(t, RL, Roff, Ron, Vin_arr, Vth, Vhold, Cm):
        Vm = np.zeros_like(t)
        Rt = np.zeros_like(t)
        tau_r = parallel(RL, Roff) * Cm
        tau_f = parallel(RL, Ron) * Cm
        charging = True
        Vm[0], Rt[0] = 0, Roff

        for i in range(1, len(t)):
            Vin = Vin_arr[i]
            Vth_l = np.sign(Vin) * Vth
            Vhold_l = np.sign(Vin) * Vhold
            if charging:
                dVm = ((Roff/(RL+Roff)*Vin) - Vm[i-1]) / tau_r
                Vm[i] = Vm[i-1] + dVm * (t[i]-t[i-1])
                Rt[i] = Roff
                if Vm[i] >= Vth_l:
                    charging = False
            else:
                dVm = ((Ron/(RL+Ron)*Vin) - Vm[i-1]) / tau_f
                Vm[i] = Vm[i-1] + dVm * (t[i]-t[i-1])
                Rt[i] = Ron
                if Vm[i] <= Vhold_l:
                    charging = True

        return Vm, Rt

    Vms, Rts = [], []
    for b in branches:
        Vm, Rt = Vm_and_Rt(t, b['RL'], Roff, Ron, b['Vin'], Vth, Vhold, b['Cm'])
        Vms.append(Vm)
        Rts.append(Rt)
    Vms = np.array(Vms)
    Rts = np.array(Rts)

    Vfusion = np.sum(Vms/Rts, axis=0) / (np.sum(1.0/Rts, axis=0) + 1.0/R0_fusion)
    Vm3 = gain * Vfusion
    Vm2, Rt2 = Vm_and_Rt_for_signal(t, 20000, Roff, Ron, Vm3, Vth, Vhold, 0.5e-9)
    Vout = R0_output / (Rt2 + R0_output) * Vm2

    spike_idx = np.where((Vout[1:] > 0.01) & (Vout[:-1] <= 0.05))[0] + 1
    spike_times = t[spike_idx]

    burst_recs, burst_starts = [], []
    if len(spike_times) >= 2:
        isi = np.diff(spike_times)
        edges = np.where(isi > 65e-6)[0] + 1
        starts = np.insert(edges, 0, 0)
        ends = np.append(edges-1, len(spike_times)-1)

        for idx, (s, e) in enumerate(zip(starts, ends)):
            pts = spike_times[s:e+1]
            if len(pts) < 2: continue
            burst_starts.append(pts[0])
            isi_vals = np.diff(pts)
            mean_isi = isi_vals.mean()*1e6
            ibi_j = [j for j in range(1, len(isi_vals)-1) if isi_vals[j-1] > isi_vals[j] < isi_vals[j+1]]
            ibi_ints = [(pts[j+1] - pts[j-1])*1e6 for j in ibi_j]
            mean_ibi = np.nan if not ibi_ints else np.mean(ibi_ints)

            burst_recs.append({
                'Mean ISI (μs)': mean_isi,
                'Mean IBI (μs)': mean_ibi
            })

    df = pd.DataFrame(burst_recs)

    if len(burst_starts) > 1:
        mbi_us = np.diff(burst_starts)*1e6
        mean_mbi = mbi_us.mean()
    else:
        mean_mbi = np.nan

    avg_isi = df['Mean ISI (μs)'].mean()
    avg_ibi = df['Mean IBI (μs)'].dropna().mean()

    return avg_isi, avg_ibi, mean_mbi

def simulate_from_RLs(rl1, rl2, rl3):
    fixed_Cm = [0.15e-9, 5.5e-9, 50e-9]
    fixed_Vin = 3.0
    branches = [
        {'RL': rl1, 'Cm': fixed_Cm[0], 'Vin': fixed_Vin},
        {'RL': rl2, 'Cm': fixed_Cm[1], 'Vin': fixed_Vin},
        {'RL': rl3, 'Cm': fixed_Cm[2], 'Vin': fixed_Vin},
    ]
    return simulate_neuro_circuit(branches)


avg_isi, avg_ibi, mean_mbi = simulate_from_RLs(6000, 7887.1082127 , 13548.38709677)

#7887.1082127 13548.38709677  6000.

print(f"Average ISI: {avg_isi:.2f} μs")
print(f"Average IBI: {avg_ibi:.2f} μs")
print(f"Average MBI: {mean_mbi:.2f} μs")