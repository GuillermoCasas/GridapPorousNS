# latexmk configuration file for porous_NS_with_Gridap (Root Level)

use Cwd;

# Change directory to the directory containing the source file before compiling.
# This ensures that all compilation paths remain relative to the source directory.
$do_cd = 1;

# Determine the target from the command line (e.g., path/to/foo.tex -> path/to/foo).
my $target = "";
foreach my $arg (@ARGV) {
    next if $arg =~ /^-/;
    if ($arg =~ /^(.*)\.tex$/) { $target = $1; last; }
    if (-e "$arg.tex")         { $target = $arg; last; }
}

if ($target ne "") {
    my $dir = ".";
    if ($target =~ /^(.*)\/(.*)$/) {
        $dir = $1;
    }
    
    if ($dir ne "." && -d $dir && -e "$dir/latexmkrc") {
        print "latexmkrc (root): Found local config in '$dir/'. Sourcing it...\n";
        my $orig_dir = Cwd::getcwd();
        chdir $dir;
        do "./latexmkrc";
        chdir $orig_dir;
    }
}
