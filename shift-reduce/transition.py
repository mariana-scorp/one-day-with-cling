class Transition(object):
    """
    This class defines a set of transitions which are applied to a
    configuration to get the next configuration.
    """
    # Define set of transitions
    LEFT_ARC = 'LEFTARC'
    RIGHT_ARC = 'RIGHTARC'
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'

    def __init__(self):
        raise ValueError('Do not construct this object!')

    @staticmethod
    def left_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        # 1. create a left dependency arc between the word on top of the stack
        # and the next token in the queue
        # 2. pop the stack
        if not conf.buffer or not conf.stack:
            return -1
        idx_wi = conf.stack[-1]
        idx_wj = conf.buffer[0]
        if idx_wi != 0:
            conf.stack.pop(-1)
            conf.arcs.append((idx_wj, relation, idx_wi))
        else:
            return -1

    @staticmethod
    def right_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        # 1. create a right dependency arc between the word on top of the stack
        # and the next token in the queue
        # 2. pop the queue
        # 3. add the element from the queue to the stack
        if not conf.buffer or not conf.stack:
            return -1
        idx_wi = conf.stack[-1]
        idx_wj = conf.buffer.pop(0)
        conf.stack.append(idx_wj)
        conf.arcs.append((idx_wi, relation, idx_wj))

    @staticmethod
    def reduce(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        # pop the stack, removing only its top item, as long as that item has a head
        if not conf.stack and conf.stack[-1] in [i[-1] for i in conf.arcs]:
            return -1
        conf.stack.pop()

    @staticmethod
    def shift(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        # move the word from the queue onto the stack
        if not conf.buffer:
            return -1
        conf.stack.append(conf.buffer.pop(0))
