

I0 = imread("30.bmp");
d2pi = 1;
I1 = double(I0(:,:,d2pi)); 
lamb_3 = [0.000650 0.000520 0.000450];
lamb_I = lamb_3(d2pi);

[r,c] = size(I1); 
Amp = zeros(r*c,1);
xyDis = zeros(r*c,2);
I1_rand = I1.*exp(1j*rand(r,c)*2*pi);
I2 = zeros(r,c);

spd = 0.1; %物面抽样距离,
Lox=spd*c;
Loy=spd*r;
xo_shift3 = [0 0 0]; 
yo_shift3 = [0 0 0]; 
xo = linspace(-Lox/2+xo_shift3(d2pi)+spd,Lox/2+xo_shift3(d2pi), c);
yo = linspace(-Loy/2+yo_shift3(d2pi)+spd,Loy/2+yo_shift3(d2pi), r);
[xo,yo] = meshgrid(xo,yo);

num = 1;
SRGB_TH = 0; 
for i=1:r
    for n=1:c
        if I1(i,n) > SRGB_TH
            I2(i,n) = I1(i,n);          
            Amp(num) =  I1_rand(i,n);
            xyDis(num,1) = xo(i,n);
            xyDis(num,2) = yo(i,n);
            num = num+1;
        end
    end
end

o_amp = Amp(1:num-1,:);
o_xydis = xyDis(1:num-1,:);

[o_num,amp_num] = size(o_amp);
lamb = lamb_I;        %激光波长，单位mm
k = 2 * pi / lamb;

zi = 30;
ze = 300;
z = zi+ze;


%===狭缝参数===================================
ls=100;
ws = 3;

%狭缝空间频率
fys = abs(1/2/lamb*(1/z-1/ze)*ls);
spds = 1/2/fys;
cs = ceil(ls/spds);
rs = ceil(ws/spds);
xs = linspace(-ls/2+spds,ls/2,cs);
ys = linspace(-ws/2+spds,ws/2,rs);
[xs,ys]=meshgrid(xs,ys);

%====全息图参数=============================================

%全息图像素间隔
spdh_normal = 3.18e-4;%全息打印机输出最大分辨率

fyh = abs( (yo(r,c)+xs(rs,cs)) / lamb / sqrt( (xo(r,c)-ys(rs,cs)).^2+(yo(r,c)+xs(rs,cs)).^2+z.^2 ) );%全息图平面最大空间频率
spdh_max = 1/2/fyh/2;
mul_spdh=floor(spdh_max/spdh_normal);
spdh = mul_spdh*spdh_normal;
%全息图点数
mul_rh = 2;
mul_ch = 4;
rh_base = 2000;
ch_base = 5000;
rh = rh_base*mul_rh;
ch = ch_base*mul_ch;

 %全息图大小
Lx=spdh*ch;  
Ly=spdh*rh;

xh = linspace(-Lx/2 + spdh,Lx/2,ch);
yh = linspace(-Ly/2 + spdh,Ly/2,rh);
% [xh,yh] = meshgrid(xh,yh);
% 生成全息图平面矩阵
for i = 1:mul_rh
    yh_tmp=zeros(rh_base);
    i_num = rh_base*(i-1)+1;
    yh_tmp = yh(i_num:i_num+rh_base-1);
    for j = 1:mul_ch
        xh_tmp=zeros(ch_base);
        j_num = ch_base*(j-1)+1;
        xh_tmp = xh(j_num:j_num+ch_base-1);
        [xh_tmpgrid,yh_tmpgrid] = meshgrid(xh_tmp,yh_tmp);   
        name_ij = [num2str(i-1),num2str(j-1)]; 
        eval(['xh',name_ij,'=xh_tmpgrid;']);   
        eval(['yh',name_ij,'=yh_tmpgrid;']); 
        %全息图物光波参数
        eval(['UF',name_ij,'=zeros(rh_base,ch_base);']);
    end
end

for onum = 1:o_num
      tic
    
      xo = o_xydis(onum,1);
      yo = o_xydis(onum,2);
      %=====线全息图参数======================
      %物点对应的线全息图中心坐标
      xh_lineCenter = ze/z*xo;
      yh_lineCenter = ze/z*yo;
      
      xh_matless = xh(xh<xh_lineCenter);%利用矩阵含有多少小于xh_lineCenter的数确定中心点在xh中位置
      yh_matless = yh(yh<yh_lineCenter);%利用矩阵含有多少小于yh_lineCenter的数确定中心点在xh中位置
      [xh_r,ch_num]= size(xh_matless);
      [yh_r,rh_num]= size(yh_matless);

      %每个点线全息图的长宽
      xh_line_len = abs((ze-z)/z)*ls;
      yh_line_wid = abs((ze-z)/z)*ws;
      %线全息图所占点数
      if mod(ceil(xh_line_len/spdh),2) 
          xh_line_dotnum = ceil(xh_line_len/spdh)+1;
      else
          xh_line_dotnum = ceil(xh_line_len/spdh);
      end
      
      if mod(ceil(yh_line_wid/spdh),2) 
          yh_line_dotnum = ceil(yh_line_wid/spdh)+1;
      else
          yh_line_dotnum = ceil(yh_line_wid/spdh);
      end 

      % =====线全息图坐标矩阵=================
      %矩阵位置
      xh_line_lnum = -xh_line_dotnum/2+ch_num;
      xh_line_hnum = xh_line_dotnum/2+ch_num-1;
      yh_line_lnum = -yh_line_dotnum/2+rh_num;
      yh_line_hnum = yh_line_dotnum/2+rh_num-1;
      
      xh_line = xh(1,xh_line_lnum:xh_line_hnum);
      yh_line = yh(1,yh_line_lnum:yh_line_hnum);
      [xh_line,yh_line]=meshgrid(xh_line,yh_line);
      
      Ux = exp(1j*k/2/(z-ze).*(xo-xh_line).^2);
      Uy = zeros(yh_line_dotnum,xh_line_dotnum);
      %计算Uo(y)
      yh_line_cuda = gpuArray(yh_line);
      e1 = exp(1j*k.*yo.^2/2/z)*(1/2/ze).*exp(1j*k*(1/2/z-1/2/ze));
      for wnum = 1:rs %rs
          Uy1 = e1.*exp(-1j*k*ys(wnum,1).^2).*ys(wnum,1).^2.*exp(1j*k.*(yh_line_cuda.*(1/ze)-yo./z).*ys(wnum,1));
          Uy = Uy+Uy1;
      end

      UF_part_cuda = o_amp(onum,1).*Uy.*Ux;
      UF_part = gather(UF_part_cuda);

%     判断线全息图矩阵在全息图矩阵的位置
      xh_lmat = floor(xh_line_lnum/ch_base);
      xh_hmat = floor(xh_line_hnum/ch_base);
      yh_lmat = floor(yh_line_lnum/rh_base);
      yh_hmat = floor(yh_line_hnum/rh_base);
      
      for i=yh_lmat:yh_hmat  
          if i == yh_lmat && i == yh_hmat
              UF_part_ylnum = yh_line_lnum;
              UF_part_yhnum = yh_line_hnum;
              UF_part_ylabs = 1;
              UF_part_yhabs = yh_line_hnum-yh_line_lnum+1;
              UFij_ylnum = mod(yh_line_lnum,rh_base);
              UFij_yhnum = mod(yh_line_hnum,rh_base);
          elseif i == yh_lmat&& i < yh_hmat
              UF_part_ylnum = yh_line_lnum;
              UF_part_yhnum = (i+1)*rh_base; 
              UF_part_ylabs = 1;
              UF_part_yhabs = UF_part_yhnum-yh_line_lnum+1;
              UFij_ylnum = mod(yh_line_lnum,rh_base);
              UFij_yhnum = rh_base;
          elseif i > yh_lmat&& i == yh_hmat
              UF_part_ylnum = i*rh_base+1;
              UF_part_yhnum = yh_line_hnum;
              UF_part_ylabs = yh_line_dotnum-UF_part_yhnum+UF_part_ylnum;
              UF_part_yhabs = yh_line_dotnum;
              UFij_ylnum = 1;
              UFij_yhnum = mod(yh_line_hnum,rh_base);
          elseif i > yh_lmat&& i < yh_hmat
              UF_part_ylnum = i*rh_base+1;
              UF_part_yhnum = (i+1)*rh_base;
%               UF_part_ylabs = (yh_lmat+1)*rh_base-yh_line_lnum+1+(i-yh_lmat-1)*rh_base;
%               UF_part_yhabs = UF_part_ylabs+rh_base-1;
              UF_part_ylabs = i*rh_base-yh_line_lnum+1;
              UF_part_yhabs = (i+1)*rh_base-yh_line_lnum;
              UFij_ylnum = 1;
              UFij_yhnum = rh_base;
          end
          for j=xh_lmat:xh_hmat
              if j == xh_lmat && j == xh_hmat
                  UF_part_xlnum = xh_line_lnum;
                  UF_part_xhnum = xh_line_hnum;
                  UF_part_xlabs = 1;
                  UF_part_xhabs = xh_line_hnum-xh_line_lnum+1;
                  UFij_xlnum = mod(xh_line_lnum,ch_base);
                  UFij_xhnum = mod(xh_line_hnum,ch_base);
              elseif j == xh_lmat&& j < xh_hmat
                  UF_part_xlnum = xh_line_lnum;
                  UF_part_xhnum = (j+1)*ch_base; 
                  UF_part_xlabs = 1;
                  UF_part_xhabs = UF_part_xhnum-xh_line_lnum+1;
                  UFij_xlnum = mod(xh_line_lnum,ch_base);
                  UFij_xhnum = ch_base;
              elseif j > xh_lmat&& j == xh_hmat
                  UF_part_xlnum = j*ch_base+1;
                  UF_part_xhnum = xh_line_hnum;
                  UF_part_xlabs = xh_line_dotnum-UF_part_xhnum+UF_part_xlnum;
                  UF_part_xhabs = xh_line_dotnum;   
                  UFij_xlnum = 1;
                  UFij_xhnum = mod(xh_line_hnum,ch_base);
              elseif j > xh_lmat&& j < xh_hmat
                  UF_part_xlnum = j*ch_base+1;
                  UF_part_xhnum = (j+1)*ch_base;
                  UF_part_xlabs = j*ch_base-xh_line_lnum+1;
                  UF_part_xhabs = (j+1)*ch_base-xh_line_lnum;
                  UFij_xlnum = 1;
                  UFij_xhnum = ch_base;
              end
%               UFij=UF_part(UF_part_ylnum:UF_part_yhnum,UF_part_xlnum:UF_part_xhnum);
%             全息的矩阵块参数
              UF_matsort_ij = ['UF',num2str(i),num2str(j),'(UFij_ylnum:UFij_yhnum,UFij_xlnum:UFij_xhnum)']; 
              eval([UF_matsort_ij,'=UF_part(UF_part_ylabs:UF_part_yhabs,UF_part_xlabs:UF_part_xhabs)+',UF_matsort_ij,';']);
              
          end
%           disp(i)
%           disp(j)
      end
    toc
    disp(onum/o_num);
end

%每个模块干涉生成全息图
for i = 1:mul_rh
    for j = 1:mul_ch
        name_ij = [num2str(i-1),num2str(j-1)]; 
        %全息图物光波参数
        eval(['R',name_ij,'=k*sind(6)*yh',name_ij,'+k*sind(0)*xh',name_ij,';']); 
        eval(['II',name_ij,'=abs(UF',name_ij,').*cos(angle(UF',name_ij,')-R',name_ij,');']);
        eval(['II',name_ij,'(II',name_ij,'>0)=255;']);
        eval(['II',name_ij,'(II',name_ij,'<=0)=0;']);
%         eval(['imwrite(II',name_ij,',"colorbin',name_ij,'.bmp");']);
    end
end

%全息图像素大小转换成全息打印机输出尺寸

for i = 1:mul_rh
    for j = 1:mul_ch
        name_ij = [num2str(i-1),num2str(j-1)]; 
        %全息图物光波参数
        eval(['IIx=uint8(II',name_ij,');']);
        II_mul = uint8(zeros(rh_base*mul_spdh,ch_base*mul_spdh));
        for iini = 1:rh_base
            i_mul = mul_spdh*(iini-1)+1;
            for jinj = 1:ch_base
                j_mul = mul_spdh*(jinj-1)+1;
                if IIx(iini,jinj)>0
                    II_mul(i_mul:i_mul+mul_spdh-1,j_mul:j_mul+mul_spdh-1)=IIx(iini,jinj);
                end
            end
        end
        eval(['imwrite(II_mul,"colorbin',name_ij,'.bmp");']);
    end
end