//
//  Prediction.h
//  TryCoreML
//
//  Created by dobby on 24/02/2018.
//  Copyright © 2018 Dobby. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import "inception_v3.h"

@interface Prediction : NSObject
- (NSString *)predictWithFoodImage:(UIImage *)foodImage;
@end
