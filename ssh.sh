#!/usr/bin/env expect
# SSH to alon with auto-password. Usage:
#   ./ssh.sh "command1 && command2"   — run commands and exit
#   ./ssh.sh                          — interactive session

set timeout 600
set host "chinmay@172.24.113.178"
set pass "Bon*Chon!White#Rice\$"

if {$argc > 0} {
    set cmd [lindex $argv 0]
    spawn ssh -o StrictHostKeyChecking=no $host
    expect "password:"
    send "$pass\r"
    expect "$ "
    send "source ~/compress/venv/bin/activate && cd ~/compress\r"
    expect "$ "
    send "$cmd\r"
    expect {
        -timeout 600 "$ " {}
        timeout { puts "\n\[TIMEOUT after 600s\]" }
    }
    send "exit\r"
    expect eof
} else {
    spawn ssh -o StrictHostKeyChecking=no $host
    expect "password:"
    send "$pass\r"
    interact
}
