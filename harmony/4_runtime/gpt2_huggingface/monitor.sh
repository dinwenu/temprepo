# 创建一个简单的监控脚本 monitor.sh
#!/bin/bash
while true; do
    echo "=== $(date) ===" >> system_monitor.log
    echo "CPU Usage:" >> system_monitor.log
    top -bn1 | head -n 12 >> system_monitor.log
    echo -e "\nMemory Usage:" >> system_monitor.log
    free -h >> system_monitor.log
    echo -e "\n\n" >> system_monitor.log
    sleep 5  # 每分钟记录一次
done

# 给脚本添加执行权限
chmod +x monitor.sh

# 在后台运行脚本
./monitor.sh &