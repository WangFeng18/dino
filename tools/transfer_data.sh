#! /bin/sh
FROM=${1}
TO=${2}
pass=a,,..fit
expect -c "
    spawn rsync -av $FROM wangfeng@10.224.28.46:$TO
    expect {
        *assword {set timeout 300; send $pass\r; exp_continue;}
    }
    exit
"
