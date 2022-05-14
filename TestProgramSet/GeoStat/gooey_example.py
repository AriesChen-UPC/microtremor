# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 5/13/2022 1:59 PM
@file: gooey_example.py
"""

from gooey import Gooey, GooeyParser


@Gooey(
    richtext_controls=True,
    program_name="VS Plot",
    encoding="utf-8",
    progress_regex=r"^progress: (\d+)%$",
)
def main():
    settings_msg = 'This program is used to plot the VS profile using Kriging method!'
    parser = GooeyParser(description=settings_msg)
    subs = parser.add_subparsers(help='commands', dest='command')
    file_select = subs.add_parser('Filepath')
    file_select.add_argument('Filepath', help="Please select the file path.", widget="DirChooser")
    file_select.add_argument('-Filename', help="Please select the data(.xls, .xlsx) for plotting.", widget="FileChooser")
    file_select.add_argument('-Loop', help="Whether a loop operation is required.", widget="Dropdown",
                             choices=['Yes', 'No'], default='No')
    param = subs.add_parser('Parameters')
    param.add_argument('-Method', help="Please select the method of the interpolation model.", widget="Dropdown",
                       choices=['spherical', 'gaussian', 'linear'], default='spherical')
    param.add_argument('-Scaling', help="Please input the Scaling Index.", widget="TextField", default=1.0)
    param.add_argument('-minColor', help="Whether to input the min color limit.", widget="TextField")
    param.add_argument('-maxColor', help="Whether to input the input the max color limit.", widget="TextField")

    args = parser.parse_args()
    print(args, flush=True)
    return args


if __name__ == '__main__':
    args = main()
    if args.command == 'Filepath':
        print(args.Filepath)
        print(args.Filename)
        print(args.Loop)
    elif args.command == 'Parameters':
        print(args.Method)
        print(args.Scaling)
        print(args.minColor)
        print(args.maxColor)
