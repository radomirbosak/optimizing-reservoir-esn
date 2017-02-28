numto = 5+5+1
#plot for [col=2:numto] 'output.dat' u 1:col w l
#set multiplot

set xlabel "Iteracie"
set ylabel "Aktivacie"

#plot 'output.dat' u 1:2 w l title "neuron 1 original", \
#	 'output.dat' u 1:7 w l title "neuron 1 perturbed", \
#	 'output.dat' u 1:12 w l title "gamma1"

plot 'output.dat' u 1:2 w l title "gamma1"

pause -1
clear
#unset multiplot
