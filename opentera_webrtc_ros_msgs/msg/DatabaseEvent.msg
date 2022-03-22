# Based on proto file: https://github.com/introlab/opentera_messages/blob/master/proto/DatabaseEvent.proto
int32 DB_UNKNOWN=0
int32 DB_CREATE=1
int32 DB_UPDATE=2
int32 DB_DELETE=3

int32 type
string object_type
string object_value
