from paddle.optimizer.lr import StepDecay, LinearWarmup

'受paddle与torch的optimizer中state_dict架构差异所限，pimm不再像timm一样自行定义Scheduler类，而是直接使用paddle自带的LRScheduler'

# 原timm.scheduler.scheduler_factory.create_scheduler
def create_scheduler(args):
    num_epochs = args.epochs
    lr_scheduler = None

    if args.sched == 'step':
        lr_scheduler = StepDecay(
            learning_rate = args.lr, 
            step_size = args.decay_epochs, 
            gamma = args.decay_rate)
        lr_scheduler = LinearWarmup(
            lr_scheduler, 
            warmup_steps = args.warmup_epochs, 
            start_lr = args.warmup_lr, 
            end_lr = args.lr)
    else:
        assert False, "暂不支持其他类型的Scheduler"

    return lr_scheduler, num_epochs
