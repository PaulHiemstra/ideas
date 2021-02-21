def add_options_to_node(tree, node, tt_data, player, remaining_options):
    for option in remaining_options:
        local_tt_data = copy.deepcopy(tt_data)           # To prevent changing these values in other branches of the tree
        local_tt_data.make_move(player, option, False)
        if node.identifier != 'root':
            new_identifier = node.identifier + option
        else:
            new_identifier = option
        tree.create_node(option, new_identifier, node.identifier, data = local_tt_data)
        if len(remaining_options) > 1 and not local_tt_data.is_endstate():
            add_options_to_node(tree, tree[new_identifier], local_tt_data, 
                                flip_player[player], remove_value_list(remaining_options, option))
    return None

