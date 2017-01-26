from nltk.compat import python_2_unicode_compatible

@python_2_unicode_compatible
class FeatureExtractor(object):
    @staticmethod
    def _check_informative(feat, underscore_is_informative=False):
        """
        Check whether a feature is informative
        """

        if feat is None:
            return False

        if feat == '':
            return False

        if not underscore_is_informative and feat == '_':
            return False

        return True

    @staticmethod
    def find_left_right_dependencies(idx, arcs):
        left_most = 1000000
        right_most = -1
        dep_left_most = ''
        dep_right_most = ''
        for (wi, r, wj) in arcs:
            if wi == idx:
                if (wj > wi) and (wj > right_most):
                    right_most = wj
                    dep_right_most = r
                if (wj < wi) and (wj < left_most):
                    left_most = wj
                    dep_left_most = r
        return dep_left_most, dep_right_most

    @staticmethod
    def extract_features(tokens, buffer, stack, arcs):
        """
        This function returns a list of string features for the classifier

        :param tokens: nodes in the dependency graph
        :param stack: partially processed words
        :param buffer: remaining input words
        :param arcs: partially built dependency tree

        :return: list(str)
        """

        result = []

        # features, according to "Dependency Parsing" by S. Kubler, R. McDonald, and J. Nivre
        if stack:
            token = tokens[stack[-1]]

            if FeatureExtractor._check_informative(token['word']):
                result.append('STK_0_FORM_' + token['word'])
            if FeatureExtractor._check_informative(token['tag']):
                result.append('STK_0_TAG_' + token['tag'])
            # the farthest child to the left and to the right, respectively
            dep_left_most, dep_right_most = FeatureExtractor.find_left_right_dependencies(stack[-1], arcs)
            if FeatureExtractor._check_informative(dep_left_most):
                result.append('STK_0_LDEP_' + dep_left_most)
            if FeatureExtractor._check_informative(dep_right_most):
                result.append('STK_0_RDEP_' + dep_right_most)

            if len(stack) > 1:
                token = tokens[stack[-2]]
                if FeatureExtractor._check_informative(token['tag']):
                    result.append('STK_1_TAG_' + token['tag'])

        if buffer:
            token = tokens[buffer[0]]

            if FeatureExtractor._check_informative(token['word']):
                result.append('BUF_0_FORM_' + token['word'])
            if FeatureExtractor._check_informative(token['tag']):
                result.append('BUF_0_TAG_' + token['tag'])
            # the farthest child to the left and to the right, respectively
            dep_left_most, dep_right_most = FeatureExtractor.find_left_right_dependencies(buffer[0], arcs)
            if FeatureExtractor._check_informative(dep_left_most):
                result.append('BUF_0_LDEP_' + dep_left_most)
            if FeatureExtractor._check_informative(dep_right_most):
                result.append('BUF_0_RDEP_' + dep_right_most)

            if len(buffer) > 1:
                token = tokens[buffer[1]]
                if FeatureExtractor._check_informative(token['word']):
                    result.append('BUF_1_FORM_' + token['word'])
                if FeatureExtractor._check_informative(token['tag']):
                    result.append('BUF_1_TAG_' + token['tag'])

            if len(buffer) > 2:
                token = tokens[buffer[2]]
                if FeatureExtractor._check_informative(token['tag']):
                    result.append('BUF_2_TAG_' + token['tag'])

            if len(buffer) > 3:
                token = tokens[buffer[3]]
                if FeatureExtractor._check_informative(token['tag']):
                    result.append('BUF_3_TAG_' + token['tag'])

        if stack and buffer:
            stack_idx0 = stack[-1]
            buffer_idx0 = buffer[0]
            result.append('DISTANCE_' + str(stack_idx0 - buffer_idx0))

        return result
