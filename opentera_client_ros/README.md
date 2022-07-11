# opentera_client_ros

 [A simple Python client](scripts/opentera_client_ros.py) connecting to the OpenTera server able to receive calls. This client works with the [opentera-teleop-service](https://github.com/introlab/opentera-teleop-service) and reacts to the following events:

* DeviceEvent : Device online/offline event
* JoinSessionEvent : Join session information event
* ParticipantEvent : Participant online/offline event
* StopSessionEvent : Stop session information event
* UserEvent : User online/offline event
* LeaveSessionEvent : User/Device/Participant leave session event
* JoinSessionReplyEvent : JoinSession acceptance event

Calls are initiated with the [opentera-teleop-service webportal](https://github.com/introlab/opentera-teleop-service/tree/main/webportal).

## Configuration

 A robot device must be created on the OpenTera server and the token must be copied in the [configuration file](config/client_config.json) in the following format :

```json
{
  "client_token": "JWT token generated from the OpenTera server",
  "url": "enter url here like https://server:port"
}
```
