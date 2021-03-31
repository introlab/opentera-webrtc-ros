# Based on proto file: https://github.com/introlab/opentera_messages/blob/master/proto/DeviceEvent.proto
int32 DEVICE_UNKNOWN=0
int32 DEVICE_CONNECTED=1
int32 DEVICE_DISCONNECTED=2
int32 DEVICE_DELETED=3
int32 DEVICE_ADDED=4
int32 DEVICE_JOINED_SESSION=5
int32 DEVICE_LEFT_SESSION=6
int32 DEVICE_STATUS_CHANGED=7
    
string device_uuid
int32 type
string device_name
string device_status
