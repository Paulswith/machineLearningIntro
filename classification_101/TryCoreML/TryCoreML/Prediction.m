//
//  Prediction.m
//  TryCoreML
//
//  Created by dobby on 24/02/2018.
//  Copyright © 2018 Dobby. All rights reserved.
//

#import "Prediction.h"
#import "UIImage+Utils.h"


#define CATOGORY_FILE @"catogory"

@interface Prediction()

@property(strong, nonatomic) NSMutableArray *catogaryArray;
@property(strong, nonatomic) inception_v3 *tfModel;

@end

@implementation Prediction
- (inception_v3 *)tfModel {
    if (!_tfModel) {
        // 1 加载模型, 本身代码会调用init的时候, 方法会调用initWithContentsOfURL, 找到inception文件进行初始化
        _tfModel = [[inception_v3 alloc] init];
    }
    return _tfModel;
}

- (NSString *)predictWithFoodImage:(UIImage *)foodImage
{
    // step1: 标准为size, 转为可传入的参数.
    UIImage *img = [foodImage scaleToSize:CGSizeMake(299, 299)];             // 转换为可传参的图片大小
    CVPixelBufferRef refImage = [[UIImage new] pixelBufferFromCGImage:img];  // 转换为可传参的类型

    
    // step2.1: 由于一开始是没有BottleneckInputPlaceholder, 直接0值初始一个传入
    MLMultiArray *holder = [[MLMultiArray alloc] initWithShape:@[@2048] dataType:MLMultiArrayDataTypeDouble error:nil];
    // step2.2: 启动预测, 预测完成后得到import__pool_3___reshape__0
    inception_v3Output *output = [self.tfModel predictionFromBottleneckInputPlaceholder__0:holder import__Mul__0:refImage error:nil];

    
    // step3: 从第二步, 完整得到了想要的BottleneckInputPlaceholder, 直接代入, 图片也代入.
    inception_v3Output *output1 = [self.tfModel predictionFromBottleneckInputPlaceholder__0:output.import__pool_3___reshape__0 import__Mul__0:refImage error:nil];
    
    
    // step4: 从final_train_ops__softMax_last__0提取预测结果
    MLMultiArray *__final = output1.final_train_ops__softMax_last__0;
    return [self poAccu:__final];
}

#pragma mark - 预测定位
- (NSString *)poAccu:(MLMultiArray *)finalAcc {
    NSInteger index = [self whoIsTheBig:finalAcc.description];
    NSString *preResult = self.catogaryArray[index];
    NSLog(@"index=%ld , predict this Image is _%@_",index,preResult);
    return preResult;
}

- (NSMutableArray *)catogaryArray{
    if (_catogaryArray.count == 0) {
        NSError *readError;
        NSString *path = [[NSBundle mainBundle] pathForResource:CATOGORY_FILE ofType:nil];
        NSString *catogoryString = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&readError];
        if (readError) {
            NSLog(@"readError : %@",readError);
        }
        NSArray *caotArray = [catogoryString componentsSeparatedByString:@"|"];
        _catogaryArray = [NSMutableArray arrayWithArray:caotArray];
    }
    return _catogaryArray;
}

- (NSInteger)whoIsTheBig:(NSString *)arrayStr {
    
    NSString *softmax = [[arrayStr componentsSeparatedByString:@"["][1] componentsSeparatedByString:@"]"][0];
    NSArray *ary = [softmax componentsSeparatedByString:@","];
    double maxNum = 0.0;
    NSInteger index = 0;
    for (int i=0; i<ary.count; i++)
    {
        double du =  [ary[i] doubleValue];
        if (du  > maxNum) {
            maxNum = du;
            index = i;
        }
    }
    NSLog(@"最大数 %e--%ld",maxNum,index);
    return index;
}

@end
