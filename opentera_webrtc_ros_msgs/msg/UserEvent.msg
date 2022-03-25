# Based on proto file: https://github.com/introlab/opentera_messages/blob/master/proto/UserEvent.proto
int32 USER_UNKNOWN=0
int32 USER_CONNECTED=1
int32 USER_DISCONNECTED=2
int32 USER_DELETED=3
int32 USER_ADDED=4
int32 USER_UPDATED=5
int32 USER_JOINED_SESSION=6
int32 USER_LEFT_SESSION=7

string user_uuid
int32 type
string user_fullname
