import scipy.constants as consts


def sort_levels(atom):
    """ Use the transition data to sort the atomic levels in energy order. """

    def get_trans(trans):
        lower = atom["transitions"][trans]["lower"]
        upper = atom["transitions"][trans]["upper"]
        dE = atom["transitions"][trans]["f0"]*consts.h
        return lower, upper, dE

    transitions = list(atom["transitions"].keys())
    sorted_levels = {}

    # use any transition as our starting point, to define an ordering for two
    # states
    lower, upper, dE = get_trans(transitions[0])
    transitions.remove(transitions[0])
    sorted_levels[lower] = {"above": upper, "above_dE": dE}
    sorted_levels[upper] = {"below": lower, "below_dE": dE}

    # order the remaining states. Each time, we add a transition that connects
    # to a state that we already have an energy order. This avoids growing
    # separate trees of states, which would have to be joined up later on
    while transitions:
        for trans in transitions:
            lower, upper, dE = get_trans(trans)
            if lower in sorted_levels.keys() and upper in sorted_levels.keys():
                raise ValueError(
                    "Transition '{}' would lead to multiply connected states, "
                    "which is not currently supported.".format(trans))
            if lower in sorted_levels.keys() or upper in sorted_levels.keys():
                break
        else:
            raise ValueError(
                "Transition '{}' would lead to a disconnected level structure,"
                " which is not currently supported.".format(trans))
        transitions.remove(trans)

        # we already have a position for the lower level in the transition, so
        # now we need to figure out where the upper level fits in
        if lower in sorted_levels.keys():
            _next = lower
            while sorted_levels[_next].get("above") is not None:
                _next_dE = sorted_levels[_next].get("above_dE")
                if _next_dE > dE:
                    break
                dE -= _next_dE
                _next = sorted_levels[_next]["above"]

            if dE < 0:
                raise ValueError("Transition '{}' gives a negative energy gap."
                                 " Did you get the upper and lower states "
                                 " the wrong way around for one transition?"
                                 .format(trans))

            sorted_levels[upper] = {"below": _next, "below_dE": dE}

            if _next is None:
                continue

            top = sorted_levels[_next].get("above")
            sorted_levels[_next]["above"] = upper
            sorted_levels[_next]["above_dE"] = dE
            if top is not None:
                sorted_levels[top]["below"] = upper
                sorted_levels[top]["below_dE"] = (
                    sorted_levels[top]["below_dE"] - dE)
                sorted_levels[upper]["above"] = top
                sorted_levels[upper]["above_dE"] = (
                    sorted_levels[top]["below_dE"])

        # otherwise, we need to fit lower in
        else:
            _prev = upper
            while sorted_levels[_prev].get("below") is not None:
                _prev_dE = sorted_levels[_prev].get("below_dE")
                if _prev_dE > dE:
                    break
                dE -= _prev_dE
                _prev = sorted_levels[_prev]["below"]

            if dE < 0:
                raise ValueError("Transition '{}' gives a negative energy gap."
                                 " Did you get the upper and lower states "
                                 " the wrong way around for one transition?"
                                 .format(trans))

            sorted_levels[lower] = {"above": _prev, "above_dE": dE}

            if _prev is None:
                continue

            bot = sorted_levels[_prev].get("below")
            sorted_levels[_prev]["below"] = lower
            sorted_levels[_prev]["below_dE"] = dE
            if bot is not None:
                sorted_levels[bot]["above"] = lower
                sorted_levels[bot]["above_dE"] = (
                    sorted_levels[bot]["above_dE"] - dE)
                assert sorted_levels[bot]["above_dE"] > 0
                sorted_levels[lower]["below"] = bot
                sorted_levels[lower]["below_dE"] = (
                    sorted_levels[bot]["above_dE"])

    # find the ground level
    ground_level = list(sorted_levels.keys())[0]
    while sorted_levels[ground_level].get("below") is not None:
        ground_level = sorted_levels[ground_level]["below"]

    # now sort levels by energy
    level = ground_level
    atom["sorted_levels"] = [{"level": level, "energy": 0}]
    E_total = 0

    while sorted_levels[level].get("above"):
        E_total += sorted_levels[level]["above_dE"]
        level = sorted_levels[level].get("above")
        atom["sorted_levels"].append({"level": level,
                                      "energy": E_total})
