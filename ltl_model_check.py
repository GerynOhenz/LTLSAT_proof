import json
from ltlf2dfa.parser.ltlf import LTLfParser, LTLfAnd, LTLfUntil, LTLfNot, LTLfAlways, LTLfAtomic, LTLfNext, LTLfOr, LTLfEventually, LTLfImplies, LTLfRelease


proof=''
proofcnt=0
printproof=False
# change: f_wait 在当前状态之前已经要求检查的cache
def checkLTL(f, f_wait, t, trace, loop_start, vocab, c={}, v=False):
    """ Checks satisfaction of a LTL formula on an execution trace

        NOTES:
        * This works by using the semantics of LTL and forward progression through recursion
        * Note that this does NOT require using any off-the-shelf planner

        ARGUMENTS:
            f       - an LTL formula (must be in TREE format using nested tuples
                      if you are using LTL dict, then use ltl['str_tree'])
            t       - time stamp where formula f is evaluated
            trace   - execution trace (a dict containing:
                        trace['name']:    trace name (have to be unique if calling from a set of traces)
                        trace['trace']:   execution trace (in propositions format)
                        trace['plan']:    plan that generated the trace (unneeded)
            vocab   - vocabulary of propositions
            c       - cache for checking LTL on subtrees
            v       - verbosity

        OUTPUT:
            satisfaction  - true/false indicating ltl satisfaction on the given trace
    """

    # change
    if t==len(trace['trace']):#循环
        t=loop_start
    global proof
    global proofcnt
    proofcnt+=1
    if printproof:
        print(proofcnt,':'+str(t)+':'+str(trace['trace'][t])+':'+str(f)+'\n')
    # proof+=str(t)+':'+str(trace['trace'][t])+':'+str(f)+'\n'
    if v:
        print('\nCurrent t = ' + str(t))
        print('Current f =', f)

    ###################################################

    # Check if first operator is a proposition
    if type(f) is str and f in vocab:
        proofcnt-=1
        return f in trace['trace'][t]
    #change: 现在公式里还有0，1
    if type(f) is str and f=='0':
        proofcnt-=1
        return False
    if type(f) is str and f=='1':
        proofcnt-=1
        return True

    # Check if sub-tree info is available in the cache
    key = (f, t, trace['name'])
    if c is not None:
        if key in c:
            if v: print('Found subtree history')
            proofcnt-=1
            return c[key]
    # change 用这个避免无限递归
    if key in f_wait:
        f_wait[key]+=1
    else:
        f_wait[key]=1

    # Check for standard logic operators
    # 语义一样不需要改
    if f[0] in ['not', '!']:
        value = not checkLTL(f[1], f_wait, t, trace, loop_start, vocab, c, v)
    elif f[0] in ['and', '&', '&&']:
        value = all((checkLTL(f[i], f_wait, t, trace, loop_start, vocab, c, v) for i in range(1, len(f))))
    elif f[0] in ['or', '||', '|']:
        value = any((checkLTL(f[i], f_wait, t, trace, loop_start, vocab, c, v) for i in range(1, len(f))))
    elif f[0] in ['imp', '->']:
        value = not (checkLTL(f[1], f_wait, t, trace, loop_start, vocab, c, v)) or checkLTL(f[2], f_wait, t, trace, loop_start, vocab, c, v)

    # change:原来这里是到了最后一个时刻进入该分支，现在改为已经有同样的调用待返回时进入该递归
    elif f_wait[key]>1: #这个已经待证明了
        # Confirm what your interpretation for this should be.
        if f[0] in ['G', 'F']:
            value = checkLTL(f[1], f_wait, t, trace, loop_start, vocab, c, v)  # Confirm what your interpretation here should be
        elif f[0] == 'U':
            # 检查 p U q，到最后时刻q还不出现则肯定没出现了，所以只要检查q
            value = checkLTL(f[2], f_wait, t, trace, loop_start, vocab, c, v)
        elif f[0] == 'W':  # weak-until
            # p W q, q可以不出现，只要p一直成立
            value = checkLTL(f[2], f_wait, t, trace, loop_start, vocab, c, v) or checkLTL(f[1], f_wait, t, trace, loop_start, vocab, c, v)
        elif f[0] == 'R':  # release (weak by default)
            # p R q，q一直为真或者有个状态 p&q为真 所以也不用改
            value = checkLTL(f[2], f_wait, t, trace, loop_start, vocab, c, v)
        elif f[0] == 'X':
            value = checkLTL(f[1], f_wait, t+1, trace, loop_start, vocab, c, v)

        # change:没有weak next
        else:
            # Does not exist in vocab, nor any of operators
            print(f,t,vocab)
            raise Exception('LTL check - something wrong')

    else:
        # Forward progression rules
        if f[0] == 'X':  # change:删去了N
            value = checkLTL(f[1], f_wait, t + 1, trace, loop_start, vocab, c, v)
        elif f[0] == 'G':
            value = checkLTL(f[1], f_wait, t, trace, loop_start, vocab, c, v) and checkLTL(('G', f[1]), f_wait, t + 1, trace, loop_start, vocab, c, v)
        elif f[0] == 'F':
            value = checkLTL(f[1], f_wait, t, trace, loop_start, vocab, c, v) or checkLTL(('F', f[1]), f_wait, t + 1, trace, loop_start, vocab, c, v)
        elif f[0] == 'U':
            # Basically enforces f[1] has to occur for f[1] U f[2] to be valid.
            value = checkLTL(f[2], f_wait, t, trace, loop_start, vocab, c, v) or (
                        checkLTL(f[1], f_wait, t, trace, loop_start, vocab, c, v) and checkLTL(('U', f[1], f[2]), f_wait, t + 1, trace, loop_start, vocab,
                                                                           c, v))

        elif f[0] == 'W':  # weak-until
            value = checkLTL(f[2], f_wait, t, trace, loop_start, vocab, c, v) or (
                        checkLTL(f[1], f_wait, t, trace, loop_start, vocab, c, v) and checkLTL(('W', f[1], f[2]), f_wait, t + 1, trace, loop_start, vocab, c,
                                                                           v))
        elif f[0] == 'R':  # release (weak by default)
            value = checkLTL(f[2], f_wait, t, trace, loop_start, vocab, c, v) and (
                        checkLTL(f[1], f_wait, t, trace, loop_start, vocab, c, v) or checkLTL(('R', f[1], f[2]), f_wait, t + 1, trace, loop_start, vocab, c, v))
        else:
            # Does not exist in vocab, nor any of operators
            print(f, t, vocab)
            raise Exception('LTL check - something wrong')

    if v: print('Returned: ' + str(value))

    # Save result
    if c is not None and type(c) is dict:
        key = (f, t, trace['name'])
        c[key] = value  # append

    if printproof:
        print(proofcnt,':'+str(t) + ':' + str(trace['trace'][t]) + ':' + str(f) + 'is '+ str(value)+ '\n')
    proofcnt -= 1
    return value

# 将输入的原始ltl公式转换成前缀公式树
def preorder(f):
    if isinstance(f, LTLfAtomic):
        return f.s
    if isinstance(f, LTLfAnd) or isinstance(f, LTLfUntil) or isinstance(f, LTLfOr) or isinstance(f, LTLfRelease) or isinstance(f, LTLfImplies):
        result = f.operator_symbol
        result += preorder(f.formulas[0])
        result += preorder(f.formulas[1])
        return result
    if isinstance(f, LTLfNot) or isinstance(f, LTLfNext) or isinstance(f, LTLfAlways) or isinstance(f, LTLfEventually):
        result = f.operator_symbol
        result += preorder(f.f)
        return result

def ltl2prefix(ltl: str):
    parser = LTLfParser()
    formula = parser(ltl.replace('1', 'true').replace('0', 'false').replace('W','R'))
    return preorder(formula).replace('true', '1').replace('false', '0').replace('R', 'W')
# 将输入的原始ltl公式转换成前缀元组
def preorder_turple(f):
    if isinstance(f, LTLfAtomic):
        return f.s.replace('true', '1').replace('false', '0')
    if isinstance(f, LTLfAnd) or isinstance(f, LTLfUntil) or isinstance(f, LTLfOr) or isinstance(f, LTLfRelease) or isinstance(f, LTLfImplies):
        return tuple([f.operator_symbol.replace('R', 'W')]+[preorder_turple(f.formulas[i]) for i in range(len(f.formulas))])
    if isinstance(f, LTLfNot) or isinstance(f, LTLfNext) or isinstance(f, LTLfAlways) or isinstance(f, LTLfEventually):
        return (f.operator_symbol,preorder_turple(f.f))

def ltl2turple(ltl: str):
    parser = LTLfParser()
    formula = parser(ltl.replace('1', 'true').replace('0', 'false').replace('W','R'))
    return preorder_turple(formula)

#将网络输出的路径转换成list
def convert_trace(trace:str):
    trace=trace.split(';')#每个状态按分号隔开
    loop_start=-1
    for i in range(0,len(trace)):
        if '{' in trace[i]:
            loop_start=i
        tstr=trace[i].replace('&','').replace('{','').replace('}','')
        trace[i]=[j for j in tstr]
    return trace,loop_start

def check(ltl_formula,trace,vocab):
    ltl = ltl_formula.replace("V", "R")
    t,loop_start=convert_trace(trace)
    trace={'name':'t','trace':t}
    formula=ltl2turple(ltl)
    if printproof:
        print('formula',formula)
    # print(ltl2prefix(ltl))
        print('trace',t)
        print('loop_start',loop_start)
    return checkLTL(formula,{},0,trace,loop_start,vocab,{})


#检查一个文件里的所有数据
def check_file_ltl(fname='t0.json'):
    global printproof
    cnt=0
    errcnt=0
    printproof=False
    with open('t0.json', 'r') as f:
        datas = json.load(f)
        for data in datas:
            cnt += 1
            # proof=''
            if not check(data['ltl'], data['trace'], vocab):
                errcnt += 1
                print(data['ltl'])
                print(data['trace'])
                # if errcnt==2:
                #     exit(0)
            # if cnt%1000==0:
            print('\rcheck %d, error %d' % (cnt, errcnt), end='')

# exit(0)
# checkdir='D:\\temps\\Teaching Temporal Logics to Neural Networks-data\\ourdata\\random_trace_nu'
# cnt=0
# errcnt=0
# for fname in os.listdir(checkdir):
#     with open(os.path.join(checkdir,fname),'r') as f:
#         datas=json.load(f)
#         for data in datas:
#             cnt+=1
#             # proof=''
#             if not check(data['ltl'],data['trace'],vocab):
#                 errcnt+=1
#                 print(data['ltl'])
#                 print(data['trace'])
#                 if errcnt==2:
#                     exit(0)
#             # if cnt%1000==0:
#             print('\rcheck %d, error %d'%(cnt,errcnt),end='')
if __name__ == "__main__":
    # 检查单个公式的
    vocab = [i for i in 'abcdefghij']
    vocab = set(vocab)
    check('((X(f)) | (h)) & ((e) U (X(j)))','{&eh;;j;&eh}',vocab)

