from common.myLogger import LogHelper

if __name__ == '__main__':
    logger = LogHelper('softwareLog')
    logger.writeLog('welecome ... ', level='info')

    # require_method = 'scene_objects_recongnition'
    # program_method = 'Program_ImageAI'

    require_method = 'single_object_recongnition'
    program_method = 'Program_ImageAI'

    if require_method == 'scene_objects_recongnition':
        if program_method == 'Program_ImageAI':
            from scene_objects_recongnition.scene_objects_recongnition import objects_recongnition
            object_test = objects_recongnition(require_method, program_method, logger)
            object_test.run()