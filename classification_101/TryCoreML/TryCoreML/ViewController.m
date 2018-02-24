//
//  ViewController.m
//  TryCoreML
//
//  Created by dobby on 22/02/2018.
//  Copyright © 2018 Dobby. All rights reserved.
//

#import "ViewController.h"
#import "Prediction.h"
#import "UIImage+Utils.h"



@interface ViewController ()<UINavigationControllerDelegate, UIImagePickerControllerDelegate>
@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (weak, nonatomic) IBOutlet UILabel *preLabel;
@property (strong, nonatomic) Prediction *prediction;
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.prediction = [Prediction new];
}
- (IBAction)addPhotoButton:(id)sender {
    [self addCamera];
}

//触发事件：拍照
- (void)addCamera
{
    UIImagePickerController *picker = [[UIImagePickerController alloc] init];
    picker.delegate = self;
    picker.allowsEditing = YES; //可编辑
    //判断是否可以打开照相机
    if ([UIImagePickerController isSourceTypeAvailable:UIImagePickerControllerSourceTypeCamera]) {
        //摄像头
        picker.sourceType = UIImagePickerControllerSourceTypeCamera;
    } else { //否则打开照片库
        picker.sourceType = UIImagePickerControllerSourceTypePhotoLibrary;
    }
    [self presentViewController:picker animated:YES completion:nil];
}


#pragma mark - UIImagePickerControllerDelegate

//拍摄完成后要执行的代理方法
- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary *)info
{
    NSString *mediaType = [info objectForKey:UIImagePickerControllerMediaType];
    if ([mediaType isEqualToString:@"public.image"]) {
        //得到照片
        UIImage *image = [info objectForKey:UIImagePickerControllerOriginalImage];
        image = [image scaleToSize:self.imageView.frame.size];
        self.imageView.image = image;
        // 异步处理, 不要占用主线程:
        dispatch_async(dispatch_queue_create(0, 0), ^{
            NSString *preString = [self.prediction predictWithFoodImage:image];
            dispatch_async(dispatch_get_main_queue(), ^{
                self.preLabel.text = preString;
            });
        });
    }
    [self dismissViewControllerAnimated:YES completion:nil];
}

//进入拍摄页面点击取消按钮
- (void)imagePickerControllerDidCancel:(UIImagePickerController *)picker
{
    [self dismissViewControllerAnimated:YES completion:nil];
}


@end
