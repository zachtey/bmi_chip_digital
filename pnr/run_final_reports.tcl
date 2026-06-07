set RPT digital_final_reports
exec mkdir -p $RPT

if {[catch {dbGet top.name} design_name] || $design_name == "0x0"} {
  restoreDesign bmi_final_clean_correct_pins.enc.dat bmi_chip_top
}

puts "Loaded design: [dbGet top.name]"
puts "Design boundary: [dbGet top.fPlan.box]"

setDesignMode -process 45
setAnalysisMode -analysisType onChipVariation -cppr both

redirect $RPT/final_pg_connectivity.rpt {
  verifyConnectivity -type special -nets {VDD VSS}
}

redirect $RPT/final_full_connectivity.rpt {
  verifyConnectivity -type all -error 1000 -warning 50
}

verify_drc -report $RPT/final_drc.rpt -limit 1000

timeDesign -postRoute -pathReports -drvReports -slackReports -numPaths 50 -prefix $RPT/setup_postroute
timeDesign -postRoute -hold -pathReports -slackReports -numPaths 50 -prefix $RPT/hold_postroute

catch {report_timing -late -max_paths 10 -path_type full_clock > $RPT/critical_setup_paths.rpt}
catch {report_timing -early -max_paths 10 -path_type full_clock > $RPT/critical_hold_paths.rpt}
catch {report_constraint -all_violators > $RPT/timing_violators.rpt}

catch {report_area > $RPT/area.rpt}
catch {reportGateCount > $RPT/gate_count.rpt}
catch {summaryReport -outfile $RPT/summary_report.rpt}

catch {reportRoute -summary > $RPT/route_summary.rpt}
catch {reportCongestion > $RPT/congestion.rpt}

catch {report_ccopt_clock_trees > $RPT/clock_tree_summary.rpt}
catch {report_clock_timing -type summary > $RPT/clock_timing_summary.rpt}

set fp [open "$RPT/design_counts.rpt" w]
puts $fp "Design name: [dbGet top.name]"
puts $fp "Design boundary: [dbGet top.fPlan.box]"
puts $fp ""
puts $fp "Top-level ports:"
puts $fp [dbGet top.terms.name]
puts $fp ""
puts $fp "Number of top-level ports: [llength [dbGet top.terms.name]]"
puts $fp "Number of instances: [llength [dbGet top.insts.name]]"
puts $fp "Number of nets: [llength [dbGet top.nets.name]]"
puts $fp ""
puts $fp "Physical pin shapes:"
puts $fp [dbGet top.terms.pins.allShapes]
close $fp

catch {setExtractRCMode -engine postRoute}
catch {extractRC}

if {[catch {report_power > $RPT/power_vectorless.rpt} err]} {
  set fp [open "$RPT/power_vectorless.rpt" w]
  puts $fp "report_power failed:"
  puts $fp $err
  close $fp
}

set fp [open "$RPT/SLIDE_SUMMARY.txt" w]
puts $fp "DIGITAL BACKEND FINAL REPORT SUMMARY"
puts $fp "===================================="
puts $fp "Design: [dbGet top.name]"
puts $fp "Boundary: [dbGet top.fPlan.box]"
puts $fp "Ports: [dbGet top.terms.name]"
puts $fp "Instances: [llength [dbGet top.insts.name]]"
puts $fp "Nets: [llength [dbGet top.nets.name]]"
puts $fp ""
puts $fp "Generated reports are in digital_final_reports/"
close $fp

puts "DONE. Reports written to: $RPT"
puts "Check summary with:"
puts "  exec cat $RPT/SLIDE_SUMMARY.txt"
