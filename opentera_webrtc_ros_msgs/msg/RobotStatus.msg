# The header will keep the timestamp
std_msgs/Header header
# Fields
bool is_charging
float32 battery_voltage
float32 battery_current
float32 battery_level  # 0 - 100
float32 cpu_usage
float32 mem_usage
float32 disk_usage
string wifi_network
float32 wifi_strength  # 0 - 100
float32 upload_speed
float32 download_speed
string local_ip
float32 mic_volume
bool is_camera_on
float32 volume
