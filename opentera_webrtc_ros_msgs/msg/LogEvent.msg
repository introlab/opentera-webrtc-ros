# Based on proto file: https://github.com/introlab/opentera_messages/blob/master/proto/LogEvent.proto
int32 LOGLEVEL_UNKNOWN=0
int32 LOGLEVEL_TRACE=7
int32 LOGLEVEL_DEBUG=6
int32 LOGLEVEL_INFO=5
int32 LOGLEVEL_WARNING=4
int32 LOGLEVEL_CRITICAL=3
int32 LOGLEVEL_ERROR=2
int32 LOGLEVEL_FATAL=1
    
int32 level
float64 timestamp
string sender
string message
