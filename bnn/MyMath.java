package helper;

import java.util.ArrayList;
import java.util.List;

public class MyMath {
    double [][]data;
    int rows,cols;

    public MyMath(int rows,int cols) {
        data= new double[rows][cols];
        this.rows=rows;
        this.cols=cols;
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {
                data[i][j]=Math.random()*2-1;
            }
        }
    }

    public void add(MyMath m)
    {
        if(cols!=m.cols || rows!=m.rows) {
            System.out.println("Shape Mismatch");
            return;
        }

        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {
                this.data[i][j]+=m.data[i][j];
            }
        }
    }

    public static MyMath fromArray(double[] A)
    {
        MyMath temp = new MyMath(A.length,1);
        for(int i = 0; i < A.length; i++)
            temp.data[i][0] = A[i];
        return temp;

    }

    public List<Double> toArray() {
        List<Double> temp= new ArrayList<>()  ;

        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {
                temp.add(data[i][j]);
            }
        }
        return temp;
    }

    public static MyMath subtract(MyMath a, MyMath b) {
        MyMath temp=new MyMath(a.rows,a.cols);
        for(int i=0;i<a.rows;i++)
        {
            for(int j=0;j<a.cols;j++)
            {
                temp.data[i][j]=a.data[i][j]-b.data[i][j];
            }
        }
        return temp;
    }

    public static MyMath transpose(MyMath a) {
        MyMath temp=new MyMath(a.cols,a.rows);
        for(int i=0;i<a.rows;i++)
        {
            for(int j=0;j<a.cols;j++)
            {
                temp.data[j][i]=a.data[i][j];
            }
        }
        return temp;
    }

    public static MyMath multiply(MyMath a, MyMath b) {
        MyMath temp=new MyMath(a.rows,b.cols);
        for(int i=0;i<temp.rows;i++)
        {
            for(int j=0;j<temp.cols;j++)
            {
                double sum=0;
                for(int k=0;k<a.cols;k++)
                {
                    sum+=a.data[i][k]*b.data[k][j];
                }
                temp.data[i][j]=sum;
            }
        }
        return temp;
    }

    public void multiply(MyMath a) {
        for(int i=0;i<a.rows;i++)
        {
            for(int j=0;j<a.cols;j++)
            {
                this.data[i][j]*=a.data[i][j];
            }
        }

    }

    public void multiply(double a) {
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {
                this.data[i][j]*=a;
            }
        }

    }

    public void sigmoid() {
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
                this.data[i][j] = 1/(1+Math.exp(-this.data[i][j]));
        }

    }

    public MyMath deriveSigmoid() {
        MyMath temp=new MyMath(rows,cols);
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
                temp.data[i][j] = this.data[i][j] * (1-this.data[i][j]);
        }
        return temp;

    }

    /*public void print()
    {
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {
                System.out.print(this.data[i][j] + " ");
            }
            System.out.println();
        }
    }*/
}
