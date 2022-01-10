# class PseudoMessage():
#     def __init__(self, pseudo_points, count, obs, receivers, message_id, t):
#         self.pseudo_points = pseudo_points
#         self.obs = obs
#         self.count = count
#         self.receivers = receivers
#         self.id = message_id
#         self.t = t

class MessageManager():
    """
    Class for keeping track of which messages to send.
    """
    def __init__(self, n_agents, agent_id, central_robot_id=-1):
        self.robot_messages = {} # {robot_id: {received_message_id: received_timestamp}}
        self.message_dict = {} # {message_id: Message}. For fast look-up of messages.
        self.received_messages = {} # id of messages that have been received. Keeps growing in time.
        # self.message_timestamps = deque()  # for expiring messages.

        self.n_agents = n_agents
        self.agent_id = agent_id
        self.central_robot_id = central_robot_id

    def add_message(self, message, t):
        """
        Add a message.
        : param t: The time stamp at which the message is added.
        """
        id = message.id
        receivers = message.receivers

        #== add self index to message receiver list==#
        receivers.append(self.agent_id)
        self.received_messages[id] = 1

        #== save to message database==#
        self.message_dict[id] = message

        for index in set(range(self.n_agents)) - set([self.agent_id]):
            if index not in receivers:
                robot_message = self.robot_messages.get(index)
                if robot_message is not None:
                    robot_message[id] = t
                else:
                    self.robot_messages[index] = {id: t}

        #== local message
        if message.receivers[0] == self.agent_id:
            central_robot_message = self.robot_messages.get(self.central_robot_id)
            if central_robot_message is not None:
                central_robot_message[id] = t
            else:
                self.robot_messages[self.central_robot_id] = {id: t}

    def retrieve_message(self, robot_id, t):
        """
        Retrieve the messages to send to a robot, up to t-1.
        :param robot_id: Id of the robot to send the message to.
        :return:
        """
        message_ids_dict = self.robot_messages.get(robot_id)
        messages = []

        ids = list(message_ids_dict.keys())
        for id in ids:
            # only retrieve previous round messages.
            if message_ids_dict[id] < t:
                # could be more elegant here...
                messages.append(self.message_dict[id])

                #== if message is sent to all robots (including central agent)
                #== remove it from dictionary ==#
                all_sent = True
                for robot_id, robot_message in self.robot_messages.items():
                    if id in robot_message:
                        all_sent = False
                if all_sent:
                    del self.message_dict[id]
                # delete message id from the robot
                del message_ids_dict[id]
        return messages

    def retrieve_local_message(self, t):
        message_ids_dict = self.robot_messages.get(self.central_robot_id)
        messages = []

        ids = list(message_ids_dict.keys())
        for id in ids:
            # only retrieve previous round messages.
            if message_ids_dict[id] < t:
                # could be more elegant here...
                messages.append(self.message_dict[id])
                # delete message id from the robot
                del message_ids_dict[id]
        return messages

    def has_received(self, message):
        return message.id in self.received_messages

    def get_num_messages(self):
        return len(self.message_dict)
