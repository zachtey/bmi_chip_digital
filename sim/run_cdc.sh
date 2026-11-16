#!/bin/tcsh
# Unit simulations of the clock-domain-crossing blocks: the request/ack
# mailbox (cdc_handshake) and the dual-clock FIFO (cdc_async_fifo). Each is
# run at several unrelated clock ratios and phases. Can be run from any
# directory.

setenv PATH ${PATH}:/vol/cadence2018/XCELIUM1809/tools/bin
setenv CDS_LIC_FILE 5280@cadencelm.eecs.northwestern.edu
setenv LM_LICENSE_FILE @cadencelm.ece.northwestern.edu

cd `dirname $0`

# source period (ps), destination period (ps), destination phase (ps)
foreach ratio ("7000 100000 1300" "100000 7000 900" "100000 100000 37000" \
               "97000 100000 0" "3000 270000 500")
    set r = ($ratio)
    xrun -64bit -access r -nokey -timescale 1ns/1ps \
        -defparam tb_cdc_handshake.SRC_PERIOD_PS=$r[1] \
        -defparam tb_cdc_handshake.DST_PERIOD_PS=$r[2] \
        -defparam tb_cdc_handshake.DST_PHASE_PS=$r[3] \
        ../rtl/cdc_sync_2ff.sv \
        ../rtl/reset_synchronizer.sv \
        ../rtl/cdc_handshake.sv \
        tb_cdc_handshake.sv
end

# write period (ps), read period (ps), read phase (ps)
foreach ratio ("100000 7000 1300" "7000 100000 900" "100000 100000 37000" \
               "100000 97000 0" "270000 3000 500")
    set r = ($ratio)
    xrun -64bit -access r -nokey -timescale 1ns/1ps \
        -defparam tb_cdc_async_fifo.WCLK_PERIOD_PS=$r[1] \
        -defparam tb_cdc_async_fifo.RCLK_PERIOD_PS=$r[2] \
        -defparam tb_cdc_async_fifo.RCLK_PHASE_PS=$r[3] \
        ../rtl/cdc_sync_2ff.sv \
        ../rtl/reset_synchronizer.sv \
        ../rtl/cdc_async_fifo.sv \
        tb_cdc_async_fifo.sv
end
