from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt; plt.ion();
import matplotlib

plt.figure()
plt.title(r'$\theta_{b}$ in each run')
plt.xlabel("Run number")
plt.ylabel(r'$\theta_{b}\ [\mu rad]$')
runs = [4372,5508,5693]
htc_eff = [67.5,67.5,62.6]
htc_eff_err = [0.3,0.2,0.4]
htc_tb = [49.3,59.3,63.7]
htc_tb_err = [0.05,0.05,0.09]

tc_eff = [55.2,55.5,45.7]
tc_eff_err = [0.2,0.2,0.2]
tc_tb = [48.5,58.6,63.9]
tc_tb_err = [0.05,0.06,0.09]

# bending plot

plt.errorbar(runs, htc_tb, fmt="-", yerr=htc_tb_err, label=r"Cut $\theta_c/2$")
plt.errorbar(runs, tc_tb, fmt="-", yerr=tc_tb_err, label=r"Cut $\theta_c$")

plt.plot(runs[:2],htc_tb[:2],"o", label=r"Pions", color='g')
plt.plot(runs[2],htc_tb[2],"o", label=r"Xenon", color='Purple')
plt.plot(runs[:2],tc_tb[:2],"o", color='g')
plt.plot(runs[2],tc_tb[2],"o", color='Purple')



plt.axvline(x=(4372+5508)/2, linestyle="dashed", color='Crimson', label="Heating")

plt.legend()
plt.savefig("img/bend_vs_time.pdf")
plt.show()


# eff plot
plt.figure()
plt.title(r'Efficiency in each run')
plt.xlabel("Run number")
plt.ylabel('Efficiency%')

plt.errorbar(runs, htc_eff, fmt="-", yerr=htc_eff_err, label=r"Cut $\theta_c/2$")
plt.errorbar(runs, tc_eff, fmt="-", yerr=tc_eff_err, label=r"Cut $\theta_c$")

plt.plot(runs[:2],htc_eff[:2],"o", label=r"Pions", color='g')
plt.plot(runs[2],htc_eff[2],"o", label=r"Xenon", color='Purple')
plt.plot(runs[:2],tc_eff[:2],"o", color='g')
plt.plot(runs[2],tc_eff[2],"o", color='Purple')


plt.axvline(x=(4372+5508)/2, linestyle="dashed", color='Crimson', label="Heating")

plt.legend()
plt.savefig("img/eff_vs_time.pdf")
plt.show()
