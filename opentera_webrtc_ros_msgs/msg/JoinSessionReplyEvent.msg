# Based on proto file: https://github.com/introlab/opentera_messages/blob/master/proto/JoinSessionReplyEvent.proto
int32 REPLY_UNKNOWN=0
int32 REPLY_ACCEPTED=1
int32 REPLY_DENIED=2
int32 REPLY_BUSY=3
int32 REPLY_TIMEOUT=4
  
string session_uuid
string user_uuid
string participant_uuid
string device_uuid
int32 join_reply
string reply_msg
