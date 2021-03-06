
# parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = "NUMBER WORDstmt_bracket : WORD '[' stmt_bracket_inside ']'stmt_bracket_inside : WORD '-' WORD\n                           | NUMBER '-' NUMBER"
    
_lr_action_items = {'WORD':([0,3,7,],[2,4,10,]),'$end':([1,8,],[0,-1,]),'[':([2,],[3,]),'NUMBER':([3,9,],[6,11,]),'-':([4,6,],[7,9,]),']':([5,10,11,],[8,-2,-3,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'stmt_bracket':([0,],[1,]),'stmt_bracket_inside':([3,],[5,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> stmt_bracket","S'",1,None,None,None),
  ('stmt_bracket -> WORD [ stmt_bracket_inside ]','stmt_bracket',4,'p_stmt_bracket','valueName.py',31),
  ('stmt_bracket_inside -> WORD - WORD','stmt_bracket_inside',3,'p_stmt_bracket_inside','valueName.py',37),
  ('stmt_bracket_inside -> NUMBER - NUMBER','stmt_bracket_inside',3,'p_stmt_bracket_inside','valueName.py',38),
]
