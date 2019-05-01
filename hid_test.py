import clarity_control as ccon
import time

cc=ccon.clarity_controller()

serialnum = cc.getSerialNumber()

print(serialnum)

cc.switchOn()

time.sleep(5) # wait for start-up

fullstat = cc.getFullStat()

print("Firmware version is ",fullstat[0])

cc.setFilterPosition(ccon.FLTPOS2)
time.sleep(0.5)
fullstat = cc.getFullStat()
print("Filter position", fullstat[4])

cc.setFilterPosition(ccon.FLTPOS4)
time.sleep(0.5)
fullstat = cc.getFullStat()
print("Filter position", fullstat[4])

cc.setDiskPosition(ccon.DSKPOS3)
time.sleep(3)
fullstat = cc.getFullStat()
print("Disk position", fullstat[3])

cc.setDiskPosition(ccon.DSKPOS2)
time.sleep(3)
fullstat = cc.getFullStat()
print("Disk position", fullstat[3])

cc.switchOff()

cc.close()


