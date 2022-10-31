import nest
import matplotlib.pyplot as plt
import numpy as np
import time
from memory_profiler import memory_usage
import pylab


class Program:
    def pois(self):
        start = time.time()
        nest.ResetKernel()
        nest.SetKernelStatus({'local_num_threads': 8})
        nest.set_verbosity('M_WARNING')
        np.set_printoptions(edgeitems=100000)
        edict = {"I_e": 100.0, "tau_m": 30.0}
        idict = {"I_e": 200.0}
        nest.CopyModel("iaf_psc_alpha", "exc_iaf_psc_alpha", params=edict)
        nest.CopyModel("iaf_psc_alpha", "inh_iaf_psc_alpha", params=idict)
        epop1 = nest.Create("exc_iaf_psc_alpha", 8 * self.x) # возбуждающий нейрон
        ipop1 = nest.Create("inh_iaf_psc_alpha", 2 * self.x) # тормозящий нейрон
        peg = nest.Create("poisson_generator", 8 * self.x) # Пуассоновский шум на возбуждающий нейрон
        pig = nest.Create("poisson_generator", 2 * self.x) # Пуассоновский шум на тормозящий
        peg[0].rate = 10000.0
        peg[1].rate = 1000.0
        pig[0].rate = 10000.0
        pig[1].rate = 1000.0
        peg.set({"start": 100.0, "stop": 200.0})
        pig.set({"start": 400.0, "stop": 500.0})
        nest.Connect(peg, epop1)
        nest.Connect(pig, ipop1)
        nest.Connect(epop1, ipop1)

        ### Первый режим (Подкл. вольтметра)
        # voltmeter = nest.Create("voltmeter")
        # nest.Connect(voltmeter, epop1)
        # nest.Connect(voltmeter, ipop1)

        ### Второй режим (Подкл. вольтметра)
        # me1 = nest.Create("multimeter", 8 * self.x)
        # mi1 = nest.Create("multimeter", 2 * self.x)
        # me1.set({"record_from": ["V_m"]})
        # mi1.set({"record_from": ["V_m"]})
        # nest.Connect(me1, epop1)
        # nest.Connect(mi1, ipop1)

        ### Симуляция
        nest.Simulate(1000.0)

        ### Первый режим (Совместное представление графиков)

        # nest.voltage_trace.from_device(voltmeter)
        # plt.show()

        ### Первый режим (Раздельное представление графиков)

        # dmm = me1.get()  # словарь с параметрами статуса
        # Vms = dmm['events'][0]['V_m']
        # ts = dmm['events'][0]['times']
        # plt.figure(1)
        # plt.title("Возбуждающие нейроны 1 популяции")
        # plt.plot(ts, Vms)
        # plt.xlabel("мс")
        # plt.ylabel("Напряжение (мембранный потенциал")
        # plt.show()
        #
        # dmm = mi1.get()  # словарь с параметрами статуса
        # Vms = dmm['events'][0]['V_m']
        # ts = dmm['events'][0]['times']
        # plt.figure(2)
        # plt.title("Тормозящие нейроны 1 популяции")
        # plt.plot(ts, Vms)
        # plt.xlabel("мс")
        # plt.ylabel("Напряжение (мембранный потенциал")
        # plt.show()
        nest.PrintNodes()
        timeval = time.time() - start
        print("Время расчета", timeval)
        return timeval

    def simple(self):
        neuron_1 = nest.Create("iaf_psc_alpha", self.x)
        voltmeter = nest.Create("voltmeter")
        neuron_1.I_e = 376.0
        nest.Connect(voltmeter, neuron_1)
        nest.Simulate(1000.0)
        nest.voltage_trace.from_device(voltmeter)
        plt.show()

    def call(self):
        num_neurons_arr = np.array([], 'float64')
        time_arr = np.array([], 'float64')
        mem = np.array([], 'float64')
        k=1000
        for n in range(1):
            num_neurons = round(k*(n + 1))
            print("Кол-во нейронов", num_neurons)
            num_neurons_arr = np.append(num_neurons_arr, num_neurons)
            self.x = round(k*(n + 1))
            time_arr = np.append(time_arr, self.pois()) # Возможна подстановка разл. функций построения нейронов (Напр. self.simple() вместо self.pois())
            mem = np.append(mem, memory_usage())
            print(mem)
        plt.figure(1)
        plt.title("График 1 популяции")
        plt.xlabel("cек")
        plt.ylabel("Кол-во нейронов")
        plt.plot(time_arr,num_neurons_arr)
        plt.show()

        plt.figure(2)
        plt.title("График 1 популяции")
        plt.xlabel("Память")
        plt.ylabel("Ко-во нейронов")
        plt.plot(mem, num_neurons_arr)
        plt.show()


if __name__ == "__main__":
    go = Program()
    go.call()