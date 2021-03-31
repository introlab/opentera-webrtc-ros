# Based on proto file: https://github.com/introlab/opentera_messages/blob/master/proto/ParticipantEvent.proto
int32 PARTICIPANT_UNKNOWN=0
int32 PARTICIPANT_CONNECTED=1
int32 PARTICIPANT_DISCONNECTED=2
int32 PARTICIPANT_DELETED=3
int32 PARTICIPANT_ADDED=4
int32 PARTICIPANT_JOINED_SESSION=5
int32 PARTICIPANT_LEFT_SESSION=6

string participant_uuid
int32 type
string participant_name
string participant_project_name
string participant_site_name
