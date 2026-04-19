#!/usr/bin/env expect
# Run a Python script on alon inside the venv. Usage:
#   ./rrun.sh "python3 -m prism compress img.png out.prism"
#   ./rrun.sh "python3 experimental/my_codec.py img.png"
# Captures stdout cleanly, strips ANSI/prompt noise.

set timeout 600
set host "chinmay@172.24.113.178"
set pass "Bon*Chon!White#Rice\$"

if {$argc < 1} {
    puts "Usage: ./rrun.sh \"command\""
    exit 1
}

set cmd [lindex $argv 0]

log_user 0
spawn ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 $host
expect "password:"
send "$pass\r"
expect "$ "
send "source ~/compress/venv/bin/activate && cd ~/compress\r"
expect "$ "
log_user 1
send "$cmd; echo __EXIT_CODE_\$?\r"
expect {
    -timeout 600 "__EXIT_CODE_*" {}
    timeout { puts "\n\[TIMEOUT\]"; exit 1 }
}
expect "$ "
log_user 0
send "exit\r"
expect eof
